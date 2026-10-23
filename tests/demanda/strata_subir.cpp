// E3 da precisão sob demanda (docs/08): trocar o nível das matrizes com o modelo carregado, entre dois tokens.
//
// Começa no Q4_K_M do Hub, mapeado do disco, e sobe as matrizes das camadas para o nível 1 e depois para o nível 2 com o
// layout misto do E2: Q4_K vira STRATA_Q4_K_S1/S2 (a base continua sendo o GGUF mapeado; os planos ficam num buffer à
// parte), Q6_K vira STRATA_Q6_K_S1 no nível 1 e STRATA_Q6_K_P2 (montado em segundo plano) no nível 2. Uma thread de
// fundo lê os planos (arquivo gerado por tests/kernel/strata_gguf_niveis.py --planos), converte e enfileira; a geração
// aplica o que estiver pronto antes de cada token (só troca tipo, ponteiros e passos do tensor).
//
//   conferir: gera token a token um texto fixo nos níveis 0 → 1 → 2 → 1 → 0, trocando tudo de uma vez, e grava os logits
//             de cada posição (para comparar bit a bit com os GGUFs n1/n2 carregados do zero, modo ref).
//   ref:      os mesmos logits com o modelo do jeito que foi carregado (n1.gguf, n2.gguf), sem troca.
//   gerar:    geração gulosa de N tokens; a subida começa no token --inicio e corre enquanto a geração continua.
//             Mede o tempo de cada token (a pausa), quando cada nível ficou completo e a velocidade em cada nível; no fim
//             desce para o nível 1 e para o 0 entre dois tokens.
//
// Compilar contra um llama.cpp com kernel/ggml/aplicar.sh (L = pasta do llama.cpp, construída em L/build):
//   gcc -O2 -march=native -ffp-contract=off -c -I kernel kernel/strata_q4kr.c kernel/strata_q4kr_avx2.c kernel/strata_separado_avx2.c
//   g++ -O2 -march=native -std=c++17 -I L/include -I L/ggml/include -I kernel tests/demanda/strata_subir.cpp \
//       strata_q4kr.o strata_q4kr_avx2.o strata_separado_avx2.o -L L/build/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,L/build/bin -lpthread -o strata_subir
//   ./strata_subir conferir q4km.gguf planos.bin texto.txt saida_prefixo [-t 4] [-n 64]
//   ./strata_subir ref      n2.gguf   -          texto.txt saida_prefixo [-t 4] [-n 64]
//   ./strata_subir gerar    q4km.gguf planos.bin texto.txt saida.csv [-t 4] [-n 400] [--inicio 16] [--alvo 2] [--nice 10]
//                           [--t-subida 3]   (threads da geração enquanto a subida corre; 0 = as mesmas)
#include "ggml.h"
#include "llama.h"
extern "C" {
#include "strata_separado.h"
}

#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static double agora() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void * aloca(size_t n) {
    void * p = aligned_alloc(64, (n + 63) / 64 * 64);
    if (!p) { fprintf(stderr, "sem memória (%zu bytes)\n", n); exit(1); }
    return p;
}

// Uma matriz com níveis: o tensor do modelo, o estado original (Q4_K/Q6_K mapeado) e os buffers de cada nível.
struct Matriz {
    std::string nome;
    bool q6 = false;
    int64_t nblocos = 0;
    int64_t off[2] = {0, 0};          // deslocamento de cada plano no arquivo de planos
    ggml_tensor * t = nullptr;
    ggml_type tipo0;
    void * data0 = nullptr;
    size_t nb0[GGML_MAX_DIMS];
    uint8_t * qh[2] = {nullptr, nullptr};   // planos na ordem qh (tipos S)
    uint8_t * p2 = nullptr;                 // Q6_K_P2 montado
    int nivel = 0;
};

static std::vector<Matriz> le_indice(const std::string & arq) {
    std::vector<Matriz> v;
    std::ifstream f(arq + ".idx");
    std::string linha;
    while (std::getline(f, linha)) {
        std::istringstream ss(linha);
        Matriz m;
        std::string grade;
        ss >> m.nome >> grade >> m.nblocos >> m.off[0];
        if (!(ss >> m.off[1])) m.off[1] = -1;
        m.q6 = grade == "Q6_K";
        if (!m.nome.empty()) v.push_back(m);
    }
    return v;
}

static void le_faixa(FILE * f, int64_t off, uint8_t * dst, size_t n) {
    if (pread(fileno(f), dst, n, off) != (ssize_t) n) { fprintf(stderr, "leitura curta nos planos\n"); exit(1); }
}

// Trabalho pronto para aplicar entre dois tokens.
struct Troca { int i, nivel; uint8_t * qh; uint8_t * p2; };

// Prepara (fora da geração) o que a matriz m precisa para chegar ao nível k, a partir do arquivo de planos.
static Troca prepara(FILE * fp, Matriz & m, int i, int k) {
    const size_t nb = (size_t) m.nblocos, bp = nb * STRATA_PLANO_BYTES;
    Troca tr{i, k, nullptr, nullptr};
    std::vector<uint8_t> cru(bp);
    le_faixa(fp, m.off[k - 1], cru.data(), bp);
    if (m.q6 && k == 2) {
        // Q6_K_P2: monta a partir do bloco R2 (base + plano 1 + plano 2, na ordem do arquivo)
        const int rb = STRATA_Q6K_BYTES + 2 * STRATA_PLANO_BYTES;
        std::vector<uint8_t> p1(bp), r(nb * rb);
        le_faixa(fp, m.off[0], p1.data(), bp);
        const uint8_t * base = (const uint8_t *) m.data0;
        for (size_t b = 0; b < nb; b++) {
            memcpy(&r[b * rb], base + b * STRATA_Q6K_BYTES, STRATA_Q6K_BYTES);
            memcpy(&r[b * rb + STRATA_Q6K_BYTES], &p1[b * 32], 32);
            memcpy(&r[b * rb + STRATA_Q6K_BYTES + 32], &cru[b * 32], 32);
        }
        tr.p2 = (uint8_t *) aloca(nb * strata_q6kp_block_bytes(2));
        strata_q6kr_to_q6kp(2, r.data(), tr.p2, m.nblocos * 256);
    } else {
        tr.qh = (uint8_t *) aloca(bp);
        strata_plano_para_qh(cru.data(), tr.qh, m.nblocos);
    }
    return tr;
}

static void passos(ggml_tensor * t, size_t bloco) {
    t->nb[0] = bloco;
    t->nb[1] = t->ne[0] / 256 * bloco;
    for (int d = 2; d < GGML_MAX_DIMS; d++) t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
}

// Aplica uma troca (só entre dois llama_decode).
static void aplica(Matriz & m, const Troca & tr) {
    ggml_tensor * t = m.t;
    if (tr.nivel == 1) {
        m.qh[0] = tr.qh;
        ggml_strata_s_planos(t, m.qh[0], nullptr);
        t->type = m.q6 ? GGML_TYPE_STRATA_Q6_K_S1 : GGML_TYPE_STRATA_Q4_K_S1;
    } else if (!m.q6) {
        m.qh[1] = tr.qh;
        ggml_strata_s_planos(t, m.qh[0], m.qh[1]);
        t->type = GGML_TYPE_STRATA_Q4_K_S2;
    } else {
        m.p2 = tr.p2;
        t->data = m.p2;
        t->type = GGML_TYPE_STRATA_Q6_K_P2;
        passos(t, strata_q6kp_block_bytes(2));
    }
    m.nivel = tr.nivel;
}

// Desce a matriz para o nível k (0 ou 1), na hora: os buffers do nível de cima ficam para liberar depois do token.
static void desce(Matriz & m, int k, std::vector<void *> & soltar) {
    if (m.nivel <= k) return;
    ggml_tensor * t = m.t;
    if (m.nivel == 2 && m.q6) {
        t->data = m.data0;
        memcpy(t->nb, m.nb0, sizeof m.nb0);
        soltar.push_back(m.p2), m.p2 = nullptr;
    }
    if (m.nivel == 2 && !m.q6) soltar.push_back(m.qh[1]), m.qh[1] = nullptr;
    if (k == 1) {
        ggml_strata_s_planos(t, m.qh[0], nullptr);
        t->type = m.q6 ? GGML_TYPE_STRATA_Q6_K_S1 : GGML_TYPE_STRATA_Q4_K_S1;
    } else {
        t->type = m.tipo0;
        if (m.qh[0]) soltar.push_back(m.qh[0]), m.qh[0] = nullptr;
    }
    m.nivel = k;
}

struct Args {
    std::string modo, modelo, planos, texto, saida;
    int threads = 4, n = 64, inicio = 16, alvo = 2, nice = 10, t_subida = 0;
};

static Args le_args(int argc, char ** argv) {
    Args a;
    if (argc < 6) {
        fprintf(stderr, "uso: %s conferir|ref|gerar MODELO PLANOS TEXTO SAIDA [-t N] [-n N] [--inicio N] [--alvo 1|2] [--nice N] [--t-subida N]\n", argv[0]);
        exit(1);
    }
    a.modo = argv[1], a.modelo = argv[2], a.planos = argv[3], a.texto = argv[4], a.saida = argv[5];
    for (int i = 6; i + 1 < argc; i += 2) {
        const std::string o = argv[i];
        const int v = atoi(argv[i + 1]);
        if (o == "-t") a.threads = v;
        else if (o == "-n") a.n = v;
        else if (o == "--inicio") a.inicio = v;
        else if (o == "--alvo") a.alvo = v;
        else if (o == "--nice") a.nice = v;
        else if (o == "--t-subida") a.t_subida = v;
        else { fprintf(stderr, "opção desconhecida %s\n", o.c_str()); exit(1); }
    }
    return a;
}

int main(int argc, char ** argv) {
    const Args a = le_args(argc, argv);
    llama_log_set([](ggml_log_level lv, const char * txt, void *) { if (lv >= GGML_LOG_LEVEL_WARN) fputs(txt, stderr); }, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;   // a base fica mapeada do GGUF publicado
    mp.use_extra_bufts = false;            // sem reempacotar os pesos: os bytes na RAM são os do arquivo
    llama_model * model = llama_model_load_from_file(a.modelo.c_str(), mp);
    if (!model) { fprintf(stderr, "não carregou %s\n", a.modelo.c_str()); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048, cp.n_batch = 512, cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = a.threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // texto → tokens
    std::ifstream ft(a.texto);
    std::string texto((std::istreambuf_iterator<char>(ft)), std::istreambuf_iterator<char>());
    texto = texto.substr(0, 200000);
    std::vector<llama_token> toks(texto.size() + 16);
    const int nt = llama_tokenize(vocab, texto.c_str(), (int) texto.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < 0) { fprintf(stderr, "tokenize falhou\n"); return 1; }
    toks.resize(nt);

    std::vector<Matriz> ms;
    FILE * fp = nullptr;
    if (a.modo != "ref") {
        ms = le_indice(a.planos);
        fp = fopen(a.planos.c_str(), "rb");
        if (!fp) { fprintf(stderr, "não abriu %s\n", a.planos.c_str()); return 1; }
        for (auto & m : ms) {
            m.t = llama_strata_tensor(model, m.nome.c_str());
            if (!m.t) { fprintf(stderr, "tensor %s não está no modelo\n", m.nome.c_str()); return 1; }
            const ggml_type esperado = m.q6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
            if (m.t->type != esperado || ggml_nelements(m.t) != m.nblocos * 256) {
                fprintf(stderr, "%s: tipo %s, %lld pesos; o índice diz outra coisa\n", m.nome.c_str(), ggml_type_name(m.t->type), (long long) ggml_nelements(m.t));
                return 1;
            }
            m.tipo0 = m.t->type, m.data0 = m.t->data;
            memcpy(m.nb0, m.t->nb, sizeof m.nb0);
        }
    }

    // um token por llama_decode, como na geração; devolve o tempo
    int pos = 0;
    auto passo = [&](llama_token tok) {
        llama_batch b = llama_batch_get_one(&tok, 1);
        const double t0 = agora();
        if (llama_decode(ctx, b)) { fprintf(stderr, "llama_decode falhou\n"); exit(1); }
        pos++;
        return agora() - t0;
    };
    auto argmax = [&]() {
        const float * lg = llama_get_logits_ith(ctx, -1);
        return (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
    };
    auto recomeca = [&]() { llama_memory_clear(llama_get_memory(ctx), true); pos = 0; };

    if (a.modo == "conferir" || a.modo == "ref") {
        // logits de a.n posições do texto, um token por vez
        auto grava = [&](const std::string & nome) {
            recomeca();
            FILE * fo = fopen(nome.c_str(), "wb");
            double tt = 0;
            for (int i = 0; i < a.n; i++) {
                tt += passo(toks[i]);
                fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), n_vocab, fo);
            }
            fclose(fo);
            printf("%s: %d posições, %.2f tokens/s\n", nome.c_str(), a.n, a.n / tt);
        };
        if (a.modo == "ref") { grava(a.saida + ".bin"); return 0; }
        std::vector<void *> soltar;
        grava(a.saida + "_n0.bin");
        for (int k = 1; k <= 2; k++) {
            const double t0 = agora();
            for (size_t i = 0; i < ms.size(); i++) aplica(ms[i], prepara(fp, ms[i], (int) i, k));
            printf("subida para o nível %d: %.2f s (leitura + conversão + troca, 1 thread)\n", k, agora() - t0);
            grava(a.saida + "_n" + std::to_string(k) + ".bin");
        }
        for (int k = 1; k >= 0; k--) {
            const double t0 = agora();
            for (auto & m : ms) desce(m, k, soltar);
            printf("descida para o nível %d: %.6f s\n", k, agora() - t0);
            grava(a.saida + "_n" + std::to_string(k) + "_desceu.bin");
            for (void * p : soltar) free(p);
            soltar.clear();
        }
        return 0;
    }

    // gerar: a subida corre numa thread de fundo enquanto a geração continua
    std::mutex mu;
    std::vector<Troca> prontas;
    std::atomic<bool> comecar{false};
    std::thread fundo([&]() {
        setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), a.nice);
        while (!comecar.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        for (int k = 1; k <= a.alvo; k++)
            for (size_t i = 0; i < ms.size(); i++) {
                Troca tr = prepara(fp, ms[i], (int) i, k);
                std::lock_guard<std::mutex> lk(mu);
                prontas.push_back(tr);
            }
    });

    FILE * fo = fopen(a.saida.c_str(), "w");
    fprintf(fo, "token,segundos,trocas,troca_us,nivel1_matrizes,nivel2_matrizes\n");
    recomeca();
    // o prompt: os primeiros 32 tokens do texto, um por vez (fora da medida)
    for (int i = 0; i < 32; i++) passo(toks[i]);
    llama_token tok = argmax();
    int n1 = 0, n2 = 0;
    double t_ini = 0, t_n1 = 0, t_n2 = 0;
    std::vector<void *> soltar;
    for (int i = 0; i < a.n; i++) {
        if (i == a.inicio) {
            comecar = true, t_ini = agora();
            if (a.t_subida) llama_set_n_threads(ctx, a.t_subida, a.t_subida);   // deixa núcleo livre para a subida
        }
        // aplica o que estiver pronto
        std::vector<Troca> lote;
        if (i < a.n - 8) { std::lock_guard<std::mutex> lk(mu); lote.swap(prontas); }
        const double ta = agora();
        for (const Troca & tr : lote) {
            aplica(ms[tr.i], tr);
            (tr.nivel == 1 ? n1 : n2)++;
        }
        const double us = 1e6 * (agora() - ta);
        if (n1 == (int) ms.size() && !t_n1) t_n1 = agora();
        if (n2 == (int) ms.size() && !t_n2) t_n2 = agora();
        if (a.t_subida && n1 + n2 == (int) ms.size() * a.alvo && i > a.inicio) llama_set_n_threads(ctx, a.threads, a.threads);
        // no fim, desce: nível 1 a 8 tokens do fim, nível 0 a 4
        if (i == a.n - 8 && n2 == (int) ms.size()) for (auto & m : ms) desce(m, 1, soltar);
        if (i == a.n - 4) for (auto & m : ms) desce(m, 0, soltar);
        const double dt = passo(tok);
        for (void * p : soltar) free(p);
        soltar.clear();
        tok = argmax();
        fprintf(fo, "%d,%.6f,%zu,%.1f,%d,%d\n", i, dt, lote.size(), us, n1, n2);
    }
    fclose(fo);
    fundo.join();
    printf("subida: nível 1 completo em %.2f s, nível 2 em %.2f s (desde o token %d)\n",
           t_n1 ? t_n1 - t_ini : -1.0, t_n2 ? t_n2 - t_ini : -1.0, a.inicio);
    return 0;
}
