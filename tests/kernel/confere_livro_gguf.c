// Confere um GGUF dos tipos STRATA_LIVRO0/1 (strata_livro.py --formato --out-tipo) contra o GGUF F16 da emulação do
// mesmo nível (--out): cada tensor do livro desquantizado pelo ggml (to_float do tipo, tabela da chave strata.livro.<nome>)
// tem de dar, depois do arredondamento para F16, exatamente os pesos da emulação.
//
//   L=llama.cpp; cc -O2 tests/kernel/confere_livro_gguf.c -I $L/ggml/include -L $L/build/bin -lggml -lggml-base -lm -Wl,-rpath,$L/build/bin -o confere_livro_gguf
//   ./confere_livro_gguf t0.gguf e0.gguf        # sai com código 1 se algum peso divergir
#include "ggml.h"
#include "gguf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char ** argv) {
    struct ggml_context * c1 = NULL, * c2 = NULL;
    struct gguf_init_params p1 = { false, &c1 }, p2 = { false, &c2 };
    struct gguf_context * g1 = gguf_init_from_file(argv[1], p1), * g2 = gguf_init_from_file(argv[2], p2);
    if (!g1 || !g2) { fprintf(stderr, "falha ao abrir\n"); return 2; }
    double pior = 0; long n_t = 0, n_dif = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(g1); i++) {
        const char * nome = gguf_get_tensor_name(g1, i);
        struct ggml_tensor * a = ggml_get_tensor(c1, nome), * b = ggml_get_tensor(c2, nome);
        if (!ggml_strata_tipo_livro(a->type)) continue;
        char chave[256]; snprintf(chave, sizeof chave, "strata.livro.%s", nome);
        const int64_t kid = gguf_find_key(g1, chave);
        int8_t tab[64] = {0};
        memcpy(tab, gguf_get_arr_data(g1, kid), gguf_get_arr_n(g1, kid));
        const int64_t n = ggml_nelements(a);
        float * fa = malloc(n * sizeof(float));
        ggml_strata_tab_usar(tab);
        ggml_get_type_traits(a->type)->to_float(a->data, fa, n);
        const ggml_fp16_t * fb = (const ggml_fp16_t *) b->data;
        double m = 0, amax = 0;
        for (int64_t j = 0; j < n; j++) {
            const float vb = ggml_fp16_to_fp32(fb[j]);
            const float va = ggml_fp32_to_fp16(fa[j]) == fb[j] ? vb : fa[j];   // igual depois do arredondamento F16?
            if (va != vb) n_dif++;
            m = fmax(m, fabs(fa[j] - vb)); amax = fmax(amax, fabs(vb));
        }
        pior = fmax(pior, m / amax); n_t++;
        free(fa);
    }
    printf("%ld tensores; pesos diferentes depois do arredondamento F16: %ld; maior diferença / maior peso: %.2e\n", n_t, n_dif, pior);
    return n_dif != 0;
}
