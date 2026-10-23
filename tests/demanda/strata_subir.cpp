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
//   g++ -O2 -march=native -std=c++17 -I L/include -I L/ggml/include -I kernel tests/demanda/strata_subir.cpp
//       strata_q4kr.o strata_q4kr_avx2.o strata_separado_avx2.o -L L/build/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,L/build/bin -lpthread -o strata_subir
//   ./strata_subir conferir q4km.gguf planos.bin texto.txt saida_prefixo [-t 4] [-n 64]
//   ./strata_subir ref      n2.gguf   -          texto.txt saida_prefixo [-t 4] [-n 64]
//   ./strata_subir gerar    q4km.gguf planos.bin texto.txt saida.csv [-t 4] [-n 400] [--inicio 16] [--alvo 2] [--nice 10]
//                           [--t-subida 3]   (threads da geração enquanto a thread de fundo trabalha; 0 = as mesmas)
//                           [--mbit 100]     (E4: os planos chegam por uma rede simulada a 100 Mbit/s)
//                           [--orcamento-ms 4] (prepara os planos na própria geração, até 4 ms entre dois tokens)
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
#include <deque>
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
    } else if (tr.qh) {
        // nível 2 separado: Q4_K_S2, ou Q6_K_S2 como passo intermediário até o Q6_K_P2 ficar montado
        m.qh[1] = tr.qh;
        ggml_strata_s_planos(t, m.qh[0], m.qh[1]);
        t->type = m.q6 ? GGML_TYPE_STRATA_Q6_K_S2 : GGML_TYPE_STRATA_Q4_K_S2;
    } else {
        m.p2 = tr.p2;
        t->data = m.p2;
        t->type = GGML_TYPE_STRATA_Q6_K_P2;
        passos(t, strata_q6kp_block_bytes(2));
        if (m.qh[1]) free(m.qh[1]), m.qh[1] = nullptr;   // o P2 não lê os planos
    }
    m.nivel = tr.nivel;
}

// Modo cooperativo: os planos chegam crus (thread da rede) e a geração os prepara em pedaços entre dois tokens. Cada
// plano vira ordem qh e entra na hora (tipos S, inclusive Q6_K_S2); depois, com o orçamento que sobrar, os Q6_K são
// montados em P2 ("polimento", só velocidade: o resultado é o mesmo bit a bit).
struct Chegada { int i, k; std::vector<uint8_t> cru, p1; };
// Buffers reservados (e tocados) antes da geração, um por matriz e nível: numa VM cada falta de página custa caro, e
// preparar um plano em memória nova chegou a ser 10× mais lento que em memória já tocada.
struct Reserva { uint8_t * qh[2] = {nullptr, nullptr}; uint8_t * p2 = nullptr; };
static std::vector<Reserva> g_reserva;

struct Trabalho {
    Chegada c;
    bool polir = false;
    uint8_t * dst = nullptr;
    int64_t feito = 0;
};
static const int64_t PEDACO_S = 2048, PEDACO_P2 = 256;   // blocos por pedaço (512 K e 64 K pesos)
static double g_t_prep[2] = {0, 0}, g_w_prep[2] = {0, 0};   // tempo e pesos preparados: [0] planos S, [1] Q6_K_P2

// Avança o trabalho até o prazo; devolve true quando a matriz ficou pronta (e a troca é aplicada).
static bool avanca(Trabalho & w, Matriz & m, double prazo, std::vector<uint8_t> & rtmp) {
    const bool p2 = w.polir;
    const int rb = STRATA_Q6K_BYTES + 2 * STRATA_PLANO_BYTES, pb = strata_q6kp_block_bytes(2);
    if (!w.dst) {
        Reserva & r = g_reserva[w.c.i];
        w.dst = p2 ? r.p2 : r.qh[w.c.k - 1];
        if (!w.dst) w.dst = (uint8_t *) aloca((size_t) m.nblocos * (p2 ? pb : STRATA_PLANO_BYTES));
    }
    while (w.feito < m.nblocos) {
        const int64_t n = std::min(p2 ? PEDACO_P2 : PEDACO_S, m.nblocos - w.feito), b0 = w.feito;
        const double tp = agora();
        if (p2) {
            rtmp.resize((size_t) n * rb);
            const uint8_t * base = (const uint8_t *) m.data0;
            for (int64_t b = 0; b < n; b++) {
                memcpy(&rtmp[b * rb], base + (b0 + b) * STRATA_Q6K_BYTES, STRATA_Q6K_BYTES);
                memcpy(&rtmp[b * rb + STRATA_Q6K_BYTES], &w.c.p1[(b0 + b) * 32], 32);
                memcpy(&rtmp[b * rb + STRATA_Q6K_BYTES + 32], &w.c.cru[(b0 + b) * 32], 32);
            }
            strata_q6kr_to_q6kp(2, rtmp.data(), w.dst + b0 * pb, n * 256);
        } else {
            strata_plano_para_qh(&w.c.cru[b0 * 32], w.dst + b0 * 32, n);
        }
        w.feito += n;
        g_t_prep[p2] += agora() - tp, g_w_prep[p2] += n * 256.0;
        if (w.feito < m.nblocos && agora() >= prazo) return false;
    }
    aplica(m, Troca{w.c.i, w.c.k, p2 ? nullptr : w.dst, p2 ? w.dst : nullptr});
    return true;
}

// Desce a matriz para o nível k (0 ou 1), na hora: os buffers do nível de cima ficam para liberar depois do token.
static void desce(Matriz & m, int k, std::vector<void *> & soltar) {
    if (m.nivel <= k) return;
    ggml_tensor * t = m.t;
    if (m.nivel == 2 && m.p2) {
        t->data = m.data0;
        memcpy(t->nb, m.nb0, sizeof m.nb0);
        soltar.push_back(m.p2), m.p2 = nullptr;
    }
    if (m.nivel == 2 && m.qh[1]) soltar.push_back(m.qh[1]), m.qh[1] = nullptr;
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
    double mbit = 0;   // rede simulada: os planos chegam a esta taxa (0 = sem limite, do disco)
    double orcamento_ms = 0;   // > 0: a preparação roda na própria geração, no máximo isso entre dois tokens
};

static Args le_args(int argc, char ** argv) {
    Args a;
    if (argc < 6) {
        fprintf(stderr, "uso: %s conferir|ref|gerar MODELO PLANOS TEXTO SAIDA [-t N] [-n N] [--inicio N] [--alvo 1|2] [--nice N] [--t-subida N] [--mbit X] [--orcamento-ms X]\n", argv[0]);
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
        else if (o == "--mbit") a.mbit = atof(argv[i + 1]);
        else if (o == "--orcamento-ms") a.orcamento_ms = atof(argv[i + 1]);
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
        // o passo intermediário do modo cooperativo: Q6_K no nível 2 separado (Q6_K_S2) em vez de montado (P2)
        for (auto & m : ms) {
            if (!m.q6) continue;
            const size_t bp = (size_t) m.nblocos * STRATA_PLANO_BYTES;
            std::vector<uint8_t> cru(bp);
            le_faixa(fp, m.off[1], cru.data(), bp);
            uint8_t * q = (uint8_t *) aloca(bp);
            strata_plano_para_qh(cru.data(), q, m.nblocos);
            soltar.push_back(m.p2), m.p2 = nullptr;
            m.t->data = m.data0;
            memcpy(m.t->nb, m.nb0, sizeof m.nb0);
            aplica(m, Troca{0, 2, q, nullptr});
        }
        grava(a.saida + "_n2s.bin");
        for (void * p : soltar) free(p);
        soltar.clear();
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
    std::deque<Chegada> chegadas;
    std::atomic<bool> comecar{false}, ocupado{false}, parar{false};
    std::thread fundo([&]() {
        setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), a.nice);
        while (!comecar.load() && !parar.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // rede simulada: o plano de cada matriz só pode ser usado depois de chegar inteiro, na ordem dos arquivos de
        // estrato (todos os planos do nível 1, depois os do nível 2)
        const double t0 = agora();
        double chegou = 0;
        for (int k = 1; k <= a.alvo; k++)
            for (size_t i = 0; i < ms.size() && !parar.load(); i++) {
                chegou += (double) ms[i].nblocos * STRATA_PLANO_BYTES;
                if (a.mbit > 0) {
                    const double espera = t0 + chegou * 8 / (a.mbit * 1e6) - agora();
                    if (espera > 0) std::this_thread::sleep_for(std::chrono::duration<double>(espera));
                }
                if (a.orcamento_ms > 0) {
                    Chegada c{(int) i, k, std::vector<uint8_t>(ms[i].nblocos * STRATA_PLANO_BYTES), {}};
                    le_faixa(fp, ms[i].off[k - 1], c.cru.data(), c.cru.size());
                    if (ms[i].q6 && k == 2) {   // o plano 1 já chegou antes; aqui só é lido de novo do arquivo local
                        c.p1.resize(c.cru.size());
                        le_faixa(fp, ms[i].off[0], c.p1.data(), c.p1.size());
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    chegadas.push_back(std::move(c));
                    continue;
                }
                ocupado = true;
                Troca tr = prepara(fp, ms[i], (int) i, k);
                ocupado = false;
                std::lock_guard<std::mutex> lk(mu);
                prontas.push_back(tr);
            }
    });

    if (a.orcamento_ms > 0) {
        const double t0 = agora();
        size_t tot = 0;
        g_reserva.resize(ms.size());
        for (size_t i = 0; i < ms.size(); i++) {
            const size_t bp = (size_t) ms[i].nblocos * STRATA_PLANO_BYTES;
            for (int k = 0; k < a.alvo; k++) g_reserva[i].qh[k] = (uint8_t *) aloca(bp), memset(g_reserva[i].qh[k], 0, bp), tot += bp;
            if (ms[i].q6 && a.alvo == 2) {
                const size_t n = (size_t) ms[i].nblocos * strata_q6kp_block_bytes(2);
                g_reserva[i].p2 = (uint8_t *) aloca(n), memset(g_reserva[i].p2, 0, n), tot += n;
            }
        }
        printf("reserva: %.0f MB tocados em %.2f s\n", tot / 1e6, agora() - t0);
    }

    FILE * fo = fopen(a.saida.c_str(), "w");
    fprintf(fo, "token,segundos,trocas,troca_us,nivel1_matrizes,nivel2_matrizes,t\n");
    recomeca();
    // o prompt: os primeiros 32 tokens do texto, um por vez (fora da medida)
    for (int i = 0; i < 32; i++) passo(toks[i]);
    llama_token tok = argmax();
    const double t_ger = agora();
    int n1 = 0, n2 = 0, threads_agora = a.threads;
    Trabalho trab;
    std::deque<Trabalho> polir;
    bool tem_trabalho = false;
    int n_p2 = 0, n_q6 = 0;
    double t_p2 = 0;
    for (const auto & m : ms) n_q6 += m.q6;
    std::vector<uint8_t> rtmp;
    double t_ini = 0, t_n1 = 0, t_n2 = 0;
    std::vector<void *> soltar;
    for (int i = 0; i < a.n; i++) {
        if (i == a.inicio) comecar = true, t_ini = agora();
        // aplica o que estiver pronto
        std::vector<Troca> lote;
        if (i < a.n - 8) { std::lock_guard<std::mutex> lk(mu); lote.swap(prontas); }
        const double ta = agora();
        for (const Troca & tr : lote) {
            aplica(ms[tr.i], tr);
            (tr.nivel == 1 ? n1 : n2)++;
        }
        size_t n_trocas = lote.size();
        if (a.orcamento_ms > 0 && i < a.n - 8) {
            // cooperativo: prepara o que chegou, em pedaços, até gastar o orçamento
            const double prazo = ta + a.orcamento_ms / 1000;
            while (agora() < prazo) {
                if (!tem_trabalho) {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!chegadas.empty()) {
                        trab = Trabalho{std::move(chegadas.front())};
                        chegadas.pop_front();
                    } else if (!polir.empty()) {
                        trab = std::move(polir.front());
                        polir.pop_front();
                    } else break;
                    tem_trabalho = true;
                }
                if (avanca(trab, ms[trab.c.i], prazo, rtmp)) {
                    n_trocas++;
                    if (trab.polir) n_p2++;
                    else {
                        (trab.c.k == 1 ? n1 : n2)++;
                        if (ms[trab.c.i].q6 && trab.c.k == 2) {   // fica para montar em P2 quando sobrar tempo
                            Trabalho pw{std::move(trab.c)};
                            pw.polir = true;
                            polir.push_back(std::move(pw));
                        }
                    }
                    tem_trabalho = false;
                }
            }
        }
        const double us = 1e6 * (agora() - ta);
        if (n1 == (int) ms.size() && !t_n1) t_n1 = agora();
        if (n2 == (int) ms.size() && !t_n2) t_n2 = agora();
        if (n_q6 && n_p2 == n_q6 && !t_p2) t_p2 = agora();
        // enquanto a thread de fundo trabalha (não enquanto espera a rede), a geração deixa um núcleo livre para ela
        if (a.t_subida) {
            const int nt = ocupado.load() ? a.t_subida : a.threads;
            if (nt != threads_agora) llama_set_n_threads(ctx, nt, nt), threads_agora = nt;
        }
        // no fim, desce: nível 1 a 8 tokens do fim, nível 0 a 4
        if (i == a.n - 8 && n2 == (int) ms.size()) for (auto & m : ms) desce(m, 1, soltar);
        if (i == a.n - 4) for (auto & m : ms) desce(m, 0, soltar);
        const double dt = passo(tok);
        for (void * p : soltar) free(p);
        soltar.clear();
        tok = argmax();
        // o tempo do token inclui o que foi feito antes dele (trocas e, no cooperativo, a preparação)
        fprintf(fo, "%d,%.6f,%zu,%.1f,%d,%d,%.3f\n", i, dt + us / 1e6, n_trocas, us, n1, n2, agora() - t_ger);
    }
    fclose(fo);
    parar = true;
    fundo.join();
    printf("subida: nível 1 completo em %.2f s, nível 2 em %.2f s (desde o token %d)\n",
           t_n1 ? t_n1 - t_ini : -1.0, t_n2 ? t_n2 - t_ini : -1.0, a.inicio);
    if (a.orcamento_ms > 0) {
        printf("Q6_K montados em P2 em %.2f s\n", t_p2 ? t_p2 - t_ini : -1.0);
        printf("preparação na geração: planos S %.3f ns/peso (%.0f M pesos), Q6_K_P2 %.2f ns/peso (%.0f M pesos)\n",
               1e9 * g_t_prep[0] / std::max(1.0, g_w_prep[0]), g_w_prep[0] / 1e6, 1e9 * g_t_prep[1] / std::max(1.0, g_w_prep[1]), g_w_prep[1] / 1e6);
    }
    return 0;
}
