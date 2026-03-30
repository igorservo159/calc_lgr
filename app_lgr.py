import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import sympy

st.set_page_config(page_title="LGR - 12 Passos", layout="wide")


# ============================================================
# Formatação LaTeX
# ============================================================

def poly_latex(coefs, var="s"):
    """Converte array de coeficientes numpy para string LaTeX."""
    grau = len(coefs) - 1
    partes = []
    for k, c in enumerate(coefs):
        exp = grau - k
        if abs(c) < 1e-12:
            continue
        ac = abs(c)
        if exp == 0:
            cs = f"{ac:g}"
        elif abs(ac - 1) < 1e-12:
            cs = ""
        else:
            cs = f"{ac:g}"
        if exp == 0:
            termo = cs
        elif exp == 1:
            termo = f"{cs}{var}" if cs else var
        else:
            termo = f"{cs}{var}^{{{exp}}}" if cs else f"{var}^{{{exp}}}"
        if not partes:
            partes.append(f"-{termo}" if c < 0 else termo)
        else:
            partes.append(f" - {termo}" if c < 0 else f" + {termo}")
    return "".join(partes) if partes else "0"


def cx_latex(c, dec=4):
    """Formata número complexo para LaTeX."""
    r = round(c.real, dec)
    i = round(c.imag, dec)
    if abs(i) < 1e-10:
        return f"{r:g}"
    if abs(r) < 1e-10:
        if abs(abs(i) - 1) < 1e-10:
            return "j" if i > 0 else "-j"
        return f"{i:g}j"
    sinal = "+" if i >= 0 else "-"
    return f"{r:g} {sinal} {abs(i):g}j"


def fatorado_latex(raizes, var="s"):
    """Forma fatorada do polinômio em LaTeX."""
    if len(raizes) == 0:
        return "1"
    fatores = []
    for r in raizes:
        if abs(r.imag) < 1e-8:
            rv = round(r.real, 4)
            if abs(rv) < 1e-8:
                fatores.append(var)
            elif rv > 0:
                fatores.append(f"({var} - {rv:g})")
            else:
                fatores.append(f"({var} + {abs(rv):g})")
        else:
            fatores.append(rf"\left({var} - ({cx_latex(r)})\right)")
    vistos = []
    contagem = {}
    for f in fatores:
        if f not in contagem:
            contagem[f] = 0
            vistos.append(f)
        contagem[f] += 1
    partes = []
    for f in vistos:
        n = contagem[f]
        partes.append(f if n == 1 else f"{f}^{{{n}}}")
    return "".join(partes)


def formatar_complexo(c):
    """Formato texto simples (uso interno)."""
    r = round(c.real, 4)
    i = round(c.imag, 4)
    if abs(i) < 1e-10:
        return f"{r}"
    s = "+" if i >= 0 else "-"
    return f"{r} {s} {abs(i)}j"


# ============================================================
# Funções computacionais
# ============================================================

def parse_coefs(texto):
    try:
        vals = [float(x) for x in texto.strip().split()]
        if len(vals) == 0:
            return None
        return np.array(vals)
    except Exception:
        return None


def separar_jw(coefs):
    grau = len(coefs) - 1
    re = {}
    im = {}
    for k, c in enumerate(coefs):
        pot = grau - k
        r = pot % 4
        if r == 0:
            re[pot] = re.get(pot, 0) + c
        elif r == 1:
            im[pot] = im.get(pot, 0) + c
        elif r == 2:
            re[pot] = re.get(pot, 0) - c
        else:
            im[pot] = im.get(pot, 0) - c

    def montar(d):
        if not d:
            return np.array([0.0])
        g = max(d.keys())
        arr = np.zeros(g + 1)
        for p, v in d.items():
            arr[g - p] = v
        return arr

    return montar(re), montar(im)


def fazer_passo1(nG, dG, nH, dH):
    num = np.convolve(nG, nH)
    den = np.convolve(dG, dH)
    n = max(len(num), len(den))
    num = np.pad(num, (n - len(num), 0))
    den = np.pad(den, (n - len(den), 0))
    return num, den


def achar_segmentos_eixo_real(zeros, polos):
    reais = []
    for p in polos:
        if abs(p.imag) < 1e-8:
            reais.append(p.real)
    for z in zeros:
        if abs(z.imag) < 1e-8:
            reais.append(z.real)
    if not reais:
        return []
    fronteiras = sorted(set(round(r, 8) for r in reais), reverse=True)
    segs = []
    for i in range(len(fronteiras) - 1):
        meio = (fronteiras[i] + fronteiras[i + 1]) / 2
        cont = sum(1 for r in reais if r > meio + 1e-10)
        if cont % 2 == 1:
            segs.append((fronteiras[i + 1], fronteiras[i]))
    if len(reais) % 2 == 1:
        segs.append((-np.inf, fronteiras[-1]))
    return segs


def calcular_assintotas(zeros, polos):
    np_ = len(polos)
    nz = len(zeros)
    diff = np_ - nz
    if diff == 0:
        return None, []
    sigma = (np.sum(polos).real - np.sum(zeros).real) / diff
    angs = [(2 * q + 1) * 180.0 / diff for q in range(diff)]
    return sigma, angs


def achar_breakaway(num, den, polos, zeros):
    dN = np.polyder(num)
    dD = np.polyder(den)
    eq = np.polysub(np.convolve(num, dD), np.convolve(den, dN))
    raizes = np.roots(eq)
    reais_pz = []
    for p in polos:
        if abs(p.imag) < 1e-8:
            reais_pz.append(p.real)
    for z in zeros:
        if abs(z.imag) < 1e-8:
            reais_pz.append(z.real)
    pts = []
    for r in raizes:
        vn = np.polyval(num, r)
        Kv = -np.polyval(den, r) / vn if abs(vn) > 1e-12 else None
        if abs(r.imag) < 1e-6:
            rr = r.real
            cont = sum(1 for x in reais_pz if x > rr + 1e-10)
            no_lgr = cont % 2 == 1
            Kr = Kv.real if Kv is not None else np.inf
            if no_lgr and Kr > 0:
                pts.append((rr, Kr))
        else:
            if Kv is not None and abs(Kv.imag) < 1e-6 and Kv.real > 0:
                pts.append((complex(r), Kv.real))
    return pts


def cruzamento_jw(den, num):
    Re_D, Im_D = separar_jw(den)
    Re_N, Im_N = separar_jw(num)
    cross = np.polysub(np.convolve(Re_D, Im_N), np.convolve(Im_D, Re_N))
    while len(cross) > 1 and abs(cross[0]) < 1e-12:
        cross = cross[1:]
    info_jw = {
        "Re_D": Re_D, "Im_D": Im_D,
        "Re_N": Re_N, "Im_N": Im_N,
        "cross": cross,
    }
    if len(cross) <= 1:
        return [], info_jw
    ws = np.roots(cross)
    resultado = []
    for w in ws:
        if abs(w.imag) > 1e-6 or w.real < 1e-8:
            continue
        omega = w.real
        ImN = np.polyval(Im_N, omega)
        ImD = np.polyval(Im_D, omega)
        ReN = np.polyval(Re_N, omega)
        ReD = np.polyval(Re_D, omega)
        if abs(ImN) > 1e-12:
            K = -ImD / ImN
        elif abs(ReN) > 1e-12:
            K = -ReD / ReN
        else:
            continue
        if K > 1e-10:
            if not any(abs(K - kk) < 1e-4 and abs(omega - ww) < 1e-4
                       for kk, ww in resultado):
                resultado.append((K, omega))
    return resultado, info_jw


def tabela_routh(den, num):
    K = sympy.Symbol('K', positive=True)
    n = max(len(den), len(num))
    dp = np.pad(den, (n - len(den), 0))
    np_ = np.pad(num, (n - len(num), 0))

    def sym(c):
        r = round(c)
        return sympy.Integer(r) if abs(c - r) < 1e-10 else sympy.nsimplify(c, rational=True)

    coefs = [sym(d) + K * sym(nn) for d, nn in zip(dp, np_)]
    grau = len(coefs) - 1
    cols = (grau + 2) // 2

    tab = [[sympy.S.Zero] * cols for _ in range(grau + 1)]
    for j in range(cols):
        if 2 * j < len(coefs):
            tab[0][j] = coefs[2 * j]
        if 2 * j + 1 < len(coefs):
            tab[1][j] = coefs[2 * j + 1]

    for i in range(2, grau + 1):
        piv = tab[i - 1][0]
        if piv == 0:
            break
        for j in range(cols - 1):
            n_ = tab[i - 1][0] * tab[i - 2][j + 1] - tab[i - 2][0] * tab[i - 1][j + 1]
            tab[i][j] = sympy.simplify(n_ / piv)

    conds = []
    k_crit = set()
    for i in range(grau + 1):
        e = sympy.simplify(tab[i][0])
        if e.has(K):
            try:
                c = sympy.solve(e > 0, K)
                if c is True or c == sympy.S.true:
                    cond_ltx = r"\forall\; K > 0"
                else:
                    cond_ltx = sympy.latex(c)
            except Exception:
                cond_ltx = None
            conds.append((grau - i, e, cond_ltx))
            sols = sympy.solve(sympy.Eq(e, 0), K)
            for s_val in sols:
                if s_val.is_real and s_val > 0:
                    k_crit.add(float(s_val))

    return {
        'tab': tab, 'grau': grau, 'cols': cols,
        'conds': conds, 'k_crits': sorted(k_crit), 'K_sym': K,
    }


def ordenar_raizes(prev, curr):
    n = len(curr)
    if n == 0:
        return curr
    out = np.zeros(n, dtype=complex)
    usado = set()
    for i in range(n):
        melhor = None
        dist_min = np.inf
        for j in range(n):
            if j not in usado:
                d = abs(prev[i] - curr[j])
                if d < dist_min:
                    dist_min = d
                    melhor = j
        out[i] = curr[melhor]
        usado.add(melhor)
    return out


def calcular_lgr(num, den, Kmax=None):
    np_ = len(den) - 1
    if Kmax is None:
        Kmax = 100.0
        for kt in [100, 500, 1000, 5000]:
            poly = np.polyadd(den, kt * num)
            rr = np.roots(poly)
            if np.max(np.abs(rr)) > 50:
                Kmax = kt
                break
        else:
            Kmax = 1000.0
    k1 = np.linspace(0, 0.1, 200)
    k2 = np.logspace(-1, np.log10(Kmax), 4800)
    Ks = np.unique(np.concatenate([k1, k2]))
    Ks.sort()
    raizes = np.zeros((len(Ks), np_), dtype=complex)
    for i, k in enumerate(Ks):
        poly = np.polyadd(den, k * num)
        r = np.roots(poly)
        if i > 0:
            r = ordenar_raizes(raizes[i - 1], r)
        raizes[i] = r
    return Ks, raizes


def testar_angulo(s_teste, zeros, polos):
    ap = sum(np.degrees(np.angle(s_teste - p)) for p in polos)
    az = sum(np.degrees(np.angle(s_teste - z)) for z in zeros)
    ang = az - ap
    ang_norm = ((ang + 180) % 360) - 180
    pertence = abs(abs(ang_norm) - 180) < 5.0
    K_val = None
    if pertence:
        prod_p = np.prod([abs(s_teste - p) for p in polos]) if len(polos) > 0 else 1.0
        prod_z = np.prod([abs(s_teste - z) for z in zeros]) if len(zeros) > 0 else 1.0
        K_val = prod_p / prod_z if prod_z > 1e-12 else np.inf
    return ang, ang_norm, pertence, K_val


def calcular_K_ponto(s_pt, zeros, polos):
    prod_p = np.prod([abs(s_pt - p) for p in polos]) if len(polos) > 0 else 1.0
    prod_z = np.prod([abs(s_pt - z) for z in zeros]) if len(zeros) > 0 else 1.0
    return prod_p / prod_z if prod_z > 1e-12 else np.inf


# ============================================================
# Funções de plotagem
# ============================================================

def limites_grafico(polos, zeros, extra_real=None):
    todos = np.concatenate([polos, zeros]) if len(zeros) > 0 else polos
    spread = max(np.ptp(todos.real), np.ptp(todos.imag), 1.0)
    marg = max(1.0, 0.5 * spread)
    x_pts = list(todos.real)
    if extra_real:
        x_pts.extend(extra_real)
    xl = (min(x_pts) - marg, max(x_pts) + marg)
    yl = max(abs(todos.imag).max() + marg * 0.5, marg)
    return xl, yl, marg


def desenhar_polos_zeros(ax, polos, zeros):
    ax.plot(polos.real, polos.imag, "rx", ms=10, mew=2, label="Polos", zorder=5)
    if len(zeros) > 0:
        ax.plot(zeros.real, zeros.imag, "go", ms=8, mew=2, fillstyle="none",
                label="Zeros", zorder=5)


def finalizar_grafico(ax, xl, yl, titulo="", limites_usr=None):
    if limites_usr is not None:
        ax.set_xlim(limites_usr[0], limites_usr[1])
        ax.set_ylim(limites_usr[2], limites_usr[3])
    else:
        ax.set_xlim(xl)
        ax.set_ylim(-yl, yl)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r"Real ($\sigma$)")
    ax.set_ylabel(r"Imaginario ($j \omega$)")
    ax.set_title(titulo)
    ax.legend(fontsize=9)


def desenhar_segmentos(ax, segs, xl):
    for i, (a, b) in enumerate(segs):
        a_plot = max(a, xl[0] - 5) if np.isfinite(a) else xl[0] - 5
        b_plot = min(b, xl[1] + 5) if np.isfinite(b) else xl[1] + 5
        lbl = "Segmento LGR" if i == 0 else ""
        ax.plot([a_plot, b_plot], [0, 0], 'b-', linewidth=4, alpha=0.6,
                solid_capstyle='round', label=lbl)


def desenhar_assintotas(ax, sigma_a, angs, xl, yl):
    line_len = max(abs(xl[0]), abs(xl[1]), yl) * 2
    for i, ang_deg in enumerate(angs):
        ang_rad = np.radians(ang_deg)
        dx = line_len * np.cos(ang_rad)
        dy = line_len * np.sin(ang_rad)
        lbl = "Assintotas" if i == 0 else ""
        ax.plot([sigma_a, sigma_a + dx], [0, dy], '--', color='darkorange',
                linewidth=1.5, alpha=0.7, label=lbl)


def desenhar_lgr_fundo(ax, todas_raizes, xl, yl):
    for j in range(todas_raizes.shape[1]):
        ramo = todas_raizes[:, j]
        ax.plot(ramo.real, ramo.imag, '-', color='gray', linewidth=1.5, alpha=0.4)

# ============================================================
# Interface Streamlit
# ============================================================

st.title("LGR - Joao Igor Ramos de Lima")
st.caption("DCA-3701 Projeto de Sistemas de Controle - UFRN")

st.markdown("""
<style>
    .katex-display { text-align: left !important; }
</style>
""", unsafe_allow_html=True)

colG, colH = st.columns(2)
with colG:
    st.latex(r"G(s) = K \cdot \frac{N_G(s)}{D_G(s)}")

col1, col2 = st.columns(2)
with col1:
    txt_nG = st.text_input("Numerador G(s)", value="1 2",
                            help="Coefs em ordem decrescente de s")
with col2:
    txt_dG = st.text_input("Denominador G(s)", value="1 4 0",
                            help="Coefs em ordem decrescente de s")

with colH:
    st.latex(r"H(s) = \frac{N_H(s)}{D_H(s)}")

col3, col4 = st.columns(2)
with col3:
    txt_nH = st.text_input("Numerador H(s)", value="1",
                            help="Coefs em ordem decrescente de s")
with col4:
    txt_dH = st.text_input("Denominador H(s)", value="1 1",
                            help="Coefs em ordem decrescente de s")

st.markdown("**Ponto de teste (Passos 11/12):**")
ct1, ct2 = st.columns(2)
with ct1:
    sr = st.number_input("Parte real", value=0.0, format="%.4f", key="sr")
with ct2:
    si = st.number_input("Parte imaginaria", value=0.0, format="%.4f", key="si")

st.markdown("**Limites dos graficos:**")
usar_limites = st.checkbox("Definir limites manualmente", value=False)
if usar_limites:
    cl1, cl2, cl3, cl4 = st.columns(4)
    with cl1:
        x_min_usr = st.number_input("x min", value=-10.0, format="%.2f", key="xmin")
    with cl2:
        x_max_usr = st.number_input("x max", value=2.0, format="%.2f", key="xmax")
    with cl3:
        y_min_usr = st.number_input("y min", value=-10.0, format="%.2f", key="ymin")
    with cl4:
        y_max_usr = st.number_input("y max", value=10.0, format="%.2f", key="ymax")

if st.button("Calcular LGR", type="primary"):
    st.session_state['calcular_lgr'] = True

if not st.session_state.get('calcular_lgr', False):
    st.stop()

nG = parse_coefs(txt_nG)
dG = parse_coefs(txt_dG)
nH = parse_coefs(txt_nH)
dH = parse_coefs(txt_dH)

if any(x is None for x in [nG, dG, nH, dH]):
    st.error("Confere os coeficientes, algo esta errado")
    st.stop()

# ============================================================
# Computacoes
# ============================================================

num, den = fazer_passo1(nG, dG, nH, dH)
zeros = np.roots(num)
polos = np.roots(den)
segs = achar_segmentos_eixo_real(zeros, polos)
ls = max(len(polos), len(zeros))
sigma_a, angs = calcular_assintotas(zeros, polos)
bk_pts = achar_breakaway(num, den, polos, zeros)
routh = tabela_routh(den, num)
cruzs, info_jw = cruzamento_jw(den, num)
Ks_lgr, todas_raizes = calcular_lgr(num, den)

xl, yl, marg = limites_grafico(polos, zeros,
                               [sigma_a] if sigma_a is not None else None)

lim_usr = (x_min_usr, x_max_usr, y_min_usr, y_max_usr) if usar_limites else None

s_test = complex(sr, si)
ang, ang_n, pert, Kp = testar_angulo(s_test, zeros, polos)
K_ponto = calcular_K_ponto(s_test, zeros, polos)

st.markdown("---")

# ============================================================
# Passo 1 - Equacao Caracteristica
# ============================================================
with st.expander("**Passo 1** - Equacao Caracteristica", expanded=True):
    nG_l = poly_latex(nG)
    dG_l = poly_latex(dG)
    nH_l = poly_latex(nH)
    dH_l = poly_latex(dH)
    num_l = poly_latex(num)
    den_l = poly_latex(den)

    st.latex(rf"G(s) = K \cdot \frac{{{nG_l}}}{{{dG_l}}}")
    st.latex(rf"H(s) = \frac{{{nH_l}}}{{{dH_l}}}")
    st.markdown("---")
    st.markdown("Funcao de transferencia de malha aberta:")
    st.latex(rf"G(s)H(s) = K \cdot \frac{{{num_l}}}{{{den_l}}} = K \cdot P(s)")
    st.markdown("Equacao caracteristica:")
    st.latex(rf"1 + K \cdot P(s) = 0 \;\;\Longrightarrow\;\; {den_l} + K\left({num_l}\right) = 0")

# ============================================================
# Passo 2 - Forma fatorada
# ============================================================
with st.expander("**Passo 2** - Forma fatorada de $P(s)$"):
    num_fat = fatorado_latex(zeros)
    den_fat = fatorado_latex(polos)
    st.markdown("Fatorando numerador e denominador de $P(s)$:")
    st.latex(rf"P(s) = \frac{{{num_fat}}}{{{den_fat}}}")

# ============================================================
# Passo 3 - Polos e zeros
# ============================================================
with st.expander("**Passo 3** - Polos e Zeros no plano $s$"):
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    desenhar_polos_zeros(ax3, polos, zeros)
    for i, p in enumerate(polos):
        ax3.annotate(rf"$p_{{{i+1}}}$", (p.real, p.imag), 
                     xytext=(8, 8), textcoords="offset points", 
                     fontsize=9, color='red')
    for i, z in enumerate(zeros):
        ax3.annotate(rf"$z_{{{i+1}}}$", (z.real, z.imag), 
                     xytext=(8, 8), textcoords="offset points", 
                     fontsize=9, color='green')
    finalizar_grafico(ax3, xl, yl, "Polos e Zeros no plano s", lim_usr)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(rf"**Polos** ($n_p = {len(polos)}$)")
        for i, p in enumerate(polos):
            st.latex(rf"p_{{{i+1}}} = {cx_latex(p)}")
    with c2:
        st.markdown(rf"**Zeros** ($n_z = {len(zeros)}$)")
        if len(zeros) > 0:
            for i, z in enumerate(zeros):
                st.latex(rf"z_{{{i+1}}} = {cx_latex(z)}")
        else:
            st.markdown("*Nenhum zero finito*")

# ============================================================
# Passo 4 - Segmentos no eixo real
# ============================================================
with st.expander("**Passo 4** - Segmentos no eixo real"):
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    desenhar_polos_zeros(ax4, polos, zeros)
    desenhar_segmentos(ax4, segs, xl)
    finalizar_grafico(ax4, xl, yl, "Segmentos do eixo real pertencentes ao LGR", lim_usr)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("**Regra:** pertencem ao LGR os segmentos do eixo real a esquerda "
                "de um numero impar de polos e zeros reais.")
    if segs:
        for a, b in segs:
            ea = f"{a:.4f}" if np.isfinite(a) else r"-\infty"
            eb = f"{b:.4f}" if np.isfinite(b) else r"+\infty"
            st.latex(rf"\left[{ea}\;,\; {eb}\right]")
    else:
        st.info("Nenhum segmento no eixo real pertence ao LGR.")

# ============================================================
# Passo 5 - Lugares separados
# ============================================================
with st.expander("**Passo 5** - Numero de lugares separados"):
    st.latex(rf"n_p = {len(polos)}, \quad n_z = {len(zeros)}")
    st.latex(rf"L_s = \max(n_p,\; n_z) = \max({len(polos)},\; {len(zeros)}) = {ls}")

# ============================================================
# Passo 6 - Simetria
# ============================================================
with st.expander("**Passo 6** - Simetria"):
    st.markdown("O LGR e **simetrico em relacao ao eixo real**, pois raizes complexas "
                "de polinomios com coeficientes reais sempre ocorrem em pares conjugados.")

# ============================================================
# Passo 7 - Assintotas
# ============================================================
with st.expander("**Passo 7** - Assintotas"):
    if sigma_a is not None:
        na = len(polos) - len(zeros)

        # --- Numero de assintotas ---
        st.markdown("**Numero de assintotas:**")
        st.latex(rf"n_a = n_p - n_z = {len(polos)} - {len(zeros)} = {na}")

        # --- Centroide: formula ---
        st.markdown("---")
        st.markdown("**Centroide (ponto de encontro das assintotas):**")
        st.latex(r"\sigma_a = \frac{\sum \operatorname{Re}(p_i) - \sum \operatorname{Re}(z_j)}{n_p - n_z}")

        # --- Centroide: soma dos polos ---
        soma_p = np.sum(polos).real
        termos_p = []
        for i, p in enumerate(polos):
            termos_p.append(f"({p.real:.4g})")
        st.markdown("Soma das partes reais dos polos:")
        st.latex(rf"\sum \operatorname{{Re}}(p_i) = {' + '.join(termos_p)} = {soma_p:.4f}")

        # --- Centroide: soma dos zeros ---
        soma_z = np.sum(zeros).real if len(zeros) > 0 else 0.0
        if len(zeros) > 0:
            termos_z = []
            for i, z in enumerate(zeros):
                termos_z.append(f"({z.real:.4g})")
            st.markdown("Soma das partes reais dos zeros:")
            st.latex(rf"\sum \operatorname{{Re}}(z_j) = {' + '.join(termos_z)} = {soma_z:.4f}")
        else:
            st.markdown("Sem zeros finitos:")
            st.latex(rf"\sum \operatorname{{Re}}(z_j) = 0")

        # --- Centroide: substituicao ---
        st.markdown("Substituindo:")
        st.latex(rf"\sigma_a = \frac{{({soma_p:.4f}) - ({soma_z:.4f})}}{{{na}}} = "
                 rf"\frac{{{soma_p - soma_z:.4f}}}{{{na}}} = {sigma_a:.4f}")

        # --- Angulos: formula ---
        st.markdown("---")
        st.markdown("**Angulos das assintotas:**")
        st.latex(rf"\phi_a = \frac{{(2q + 1) \cdot 180^\circ}}{{n_a}} = "
                 rf"\frac{{(2q + 1) \cdot 180^\circ}}{{{na}}}")

        # --- Angulos: cada q ---
        st.markdown("Calculando para cada $q$:")
        for q, a_deg in enumerate(angs):
            st.latex(rf"q = {q}: \quad \phi_a = \frac{{(2 \cdot {q} + 1) \cdot 180^\circ}}"
                     rf"{{{na}}} = \frac{{{2*q+1} \cdot 180^\circ}}{{{na}}} = {a_deg:.1f}^\circ")

        # --- Grafico ---
        st.markdown("---")
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        desenhar_polos_zeros(ax7, polos, zeros)
        desenhar_segmentos(ax7, segs, xl)
        desenhar_assintotas(ax7, sigma_a, angs, xl, yl)
        ax7.plot(sigma_a, 0, "k+", ms=12, mew=2,
                 label=rf"Centroide ($\sigma_a = {sigma_a:.2f}$)")
        finalizar_grafico(ax7, xl, yl, "LGR - Assintotas", lim_usr)
        fig7.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)
    else:
        st.latex(r"n_p = n_z \;\Rightarrow\; \text{sem assintotas}")

# ============================================================
# Passo 8 - Breakaway / Break-in
# ============================================================
with st.expander("**Passo 8** - Pontos de saida/entrada (descolamento)"):
    num_l = poly_latex(num)
    den_l = poly_latex(den)

    # --- Isolando K ---
    st.markdown("**1) Isolar $K$ na equacao caracteristica:**")
    st.latex(rf"D(s) + K \cdot N(s) = 0 \;\;\Longrightarrow\;\; K = -\frac{{D(s)}}{{N(s)}}")
    st.latex(rf"K = -\frac{{{den_l}}}{{{num_l}}}")

    # --- Condicao de descolamento ---
    st.markdown("**2) Condicao de descolamento** $\\frac{dK}{ds} = 0$:")
    st.latex(r"\frac{dK}{ds} = -\frac{D'(s) \cdot N(s) - D(s) \cdot N'(s)}{N(s)^2} = 0")
    st.markdown("Para o numerador ser zero:")
    st.latex(r"D'(s) \cdot N(s) - D(s) \cdot N'(s) = 0")

    # --- Derivadas ---
    dN = np.polyder(num)
    dD = np.polyder(den)
    st.markdown("**3) Calculando as derivadas:**")
    st.latex(rf"N(s) = {num_l}")
    st.latex(rf"N'(s) = {poly_latex(dN)}")
    st.latex(rf"D(s) = {den_l}")
    st.latex(rf"D'(s) = {poly_latex(dD)}")

    # --- Equacao resultante ---
    bk_eq = np.polysub(np.convolve(den, dN), np.convolve(num, dD))
    bk_eq_display = np.polysub(np.convolve(num, dD), np.convolve(den, dN))
    st.markdown("**4) Equacao de descolamento:**")
    st.latex(rf"{poly_latex(bk_eq_display)} = 0")

    # --- Raizes ---
    bk_raizes_all = np.roots(bk_eq_display)
    st.markdown("**5) Raizes da equacao de descolamento:**")

    reais_pz = []
    for p in polos:
        if abs(p.imag) < 1e-8:
            reais_pz.append(p.real)
    for z in zeros:
        if abs(z.imag) < 1e-8:
            reais_pz.append(z.real)

    for r in bk_raizes_all:
        vn = np.polyval(num, r)
        Kv = -np.polyval(den, r) / vn if abs(vn) > 1e-12 else None

        if abs(r.imag) < 1e-6:
            rr = r.real
            cont = sum(1 for x in reais_pz if x > rr + 1e-10)
            no_lgr = cont % 2 == 1
            Kr = Kv.real if Kv is not None else np.inf
            status = r"\text{pertence ao LGR}" if no_lgr else r"\text{fora do LGR}"
            valido = no_lgr and Kr > 0
            st.latex(rf"s = {rr:.4f}, \quad K = {Kr:.4f} \quad [{status}]")
            if valido:
                st.success(rf"Ponto de descolamento valido: $s = {rr:.4f}$ com $K = {Kr:.4f}$")
        else:
            if Kv is not None and abs(Kv.imag) < 1e-6 and Kv.real > 0:
                st.latex(rf"s = {cx_latex(r)}, \quad K = {Kv.real:.4f} \quad [\text{{no LGR}}]")
            else:
                Ks = cx_latex(Kv) if Kv is not None else r"\infty"
                st.latex(rf"s = {cx_latex(r)}, \quad K = {Ks} \quad [\text{{fora}}]")

    # --- Grafico ---
    st.markdown("---")
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    desenhar_polos_zeros(ax8, polos, zeros)
    desenhar_segmentos(ax8, segs, xl)
    if sigma_a is not None:
        desenhar_assintotas(ax8, sigma_a, angs, xl, yl)
    bk_validos = [(s_bk, k_bk) for s_bk, k_bk in bk_pts if not isinstance(s_bk, complex)]
    if bk_validos:
        bk_x = [s_bk for s_bk, _ in bk_validos]
        ax8.plot(bk_x, [0] * len(bk_x), 'md', ms=10, mew=2,
                 label="Breakaway/Break-in", zorder=6)
        for s_bk, k_bk in bk_validos:
            ax8.annotate(rf"$s={s_bk:.2f},\; K={k_bk:.2f}$", (s_bk, 0),
                         xytext=(5, 5), textcoords="offset points",
                         fontsize=8, color='purple')
    fig8.tight_layout()
    st.pyplot(fig8)
    plt.close(fig8)

# ============================================================
# Passo 9 - Cruzamento com eixo imaginario
# ============================================================
with st.expander("**Passo 9** - Cruzamento com o eixo imaginario"):
    # --- Tabela de Routh ---
    if routh is not None:
        tab = routh['tab']
        grau = routh['grau']
        cols = routh['cols']
        K_sym = routh['K_sym']

        st.markdown("**Tabela de Routh-Hurwitz:**")
        st.markdown("A partir da equacao caracteristica $D(s) + K \\cdot N(s) = 0$:")

        # Montar array LaTeX
        linhas_ltx = []
        for i in range(grau + 1):
            exp = grau - i
            cells = []
            for j in range(cols):
                elem = sympy.simplify(tab[i][j])
                cells.append(sympy.latex(elem))
            linhas_ltx.append(f"s^{{{exp}}} & " + " & ".join(cells))

        tabela_ltx = r"\begin{array}{c|" + "c" * cols + "}\n"
        tabela_ltx += r" \hline" + "\n"
        tabela_ltx += (r" \\" + "\n").join(linhas_ltx) + "\n"
        tabela_ltx += r" \\ \hline" + "\n"
        tabela_ltx += r"\end{array}"
        st.latex(tabela_ltx)

        # --- Condicoes de estabilidade ---
        if routh['conds']:
            st.markdown("**Condicoes de estabilidade** (primeira coluna $> 0$):")
            for exp, expr, cond_ltx in routh['conds']:
                expr_ltx = sympy.latex(expr)
                if cond_ltx is not None:
                    st.latex(rf"s^{{{exp}}}: \quad {expr_ltx} > 0 \;\;\Rightarrow\;\; {cond_ltx}")
                else:
                    st.latex(rf"s^{{{exp}}}: \quad {expr_ltx} > 0")

        # --- K criticos ---
        k_crits = routh['k_crits']
        if k_crits:
            st.markdown("**Valores criticos de $K$** (onde a primeira coluna se anula):")
            for kc in k_crits:
                st.latex(rf"K_{{\text{{crit}}}} = {kc:.4f}")

    # --- Metodo da substituicao s = jw ---
    st.markdown("---")
    st.markdown("**Metodo alternativo:** substituindo $s = j\\omega$ na equacao caracteristica "
                "e separando partes real e imaginaria:")
    st.latex(rf"\text{{Re}}_D(\omega) = {poly_latex(info_jw['Re_D'], var=r'\omega')}")
    st.latex(rf"\text{{Im}}_D(\omega) = {poly_latex(info_jw['Im_D'], var=r'\omega')}")
    st.latex(rf"\text{{Re}}_N(\omega) = {poly_latex(info_jw['Re_N'], var=r'\omega')}")
    st.latex(rf"\text{{Im}}_N(\omega) = {poly_latex(info_jw['Im_N'], var=r'\omega')}")

    st.markdown("Eliminando $K$ entre as equacoes real e imaginaria:")
    st.latex(rf"\text{{Re}}_D \cdot \text{{Im}}_N - \text{{Im}}_D \cdot \text{{Re}}_N = 0")
    st.latex(rf"{poly_latex(info_jw['cross'], var=r'\omega')} = 0")

    if cruzs:
        st.markdown("**Solucoes validas** ($\\omega > 0$, $K > 0$):")
        for Kc, wc in cruzs:
            # Mostrar como K foi calculado
            ImN_val = np.polyval(info_jw['Im_N'], wc)
            ImD_val = np.polyval(info_jw['Im_D'], wc)
            st.latex(rf"\omega = {wc:.4f} \;\;\Rightarrow\;\; "
                     rf"K = -\frac{{\text{{Im}}_D({wc:.4f})}}{{\text{{Im}}_N({wc:.4f})}} = "
                     rf"-\frac{{{ImD_val:.4f}}}{{{ImN_val:.4f}}} = {Kc:.4f}")
            st.latex(rf"\therefore \quad s = \pm\, {wc:.4f}\,j, \quad K = {Kc:.4f}")
    else:
        st.info("O LGR nao cruza o eixo imaginario para $K > 0$.")

    # --- Grafico ---
    st.markdown("---")
    fig9, ax9 = plt.subplots(figsize=(10, 6))
    desenhar_lgr_fundo(ax9, todas_raizes, xl, yl)
    desenhar_polos_zeros(ax9, polos, zeros)
    if cruzs:
        for Kc, wc in cruzs:
            ax9.plot(0, wc, 's', ms=10, color='cyan', markeredgecolor='navy',
                     mew=2, zorder=6,
                     label=rf"$j\omega = {wc:.2f}j\;(K={Kc:.2f})$")
            ax9.plot(0, -wc, 's', ms=10, color='cyan', markeredgecolor='navy',
                     mew=2, zorder=6)
    finalizar_grafico(ax9, xl, yl, "LGR - Cruzamento com eixo imaginario", lim_usr)
    fig9.tight_layout()
    st.pyplot(fig9)
    plt.close(fig9)

# ============================================================
# Passo 10 - Angulos de partida/chegada
# ============================================================
with st.expander("**Passo 10** - Angulos de partida e chegada"):
    polos_cx = [p for p in polos if p.imag > 1e-8]
    zeros_cx = [z for z in zeros if z.imag > 1e-8]

    if polos_cx or zeros_cx:
        # --- Angulos de partida ---
        angulos_partida = {}
        if polos_cx:
            st.markdown("### Angulos de partida (polos complexos)")
            st.markdown("**Formula:**")
            st.latex(r"\theta_d = 180^\circ - \sum_{j \neq k} \angle(p_k - p_j) "
                     r"+ \sum_j \angle(p_k - z_j)")

            for pk in polos_cx:
                st.markdown(f"---")
                st.markdown(rf"**Polo $p_k = {cx_latex(pk)}$:**")

                # Angulos dos outros polos
                ang_polos = []
                st.markdown("Angulos dos outros polos:")
                for pj in polos:
                    if abs(pj - pk) > 1e-10:
                        a = np.degrees(np.angle(pk - pj))
                        ang_polos.append(a)
                        diff = pk - pj
                        st.latex(rf"\angle(p_k - p_j) = \angle({cx_latex(pk)} - ({cx_latex(pj)})) "
                                 rf"= \angle({cx_latex(diff)}) = {a:.2f}^\circ")

                # Angulos dos zeros
                ang_zeros = []
                if len(zeros) > 0:
                    st.markdown("Angulos dos zeros:")
                    for zj in zeros:
                        a = np.degrees(np.angle(pk - zj))
                        ang_zeros.append(a)
                        diff = pk - zj
                        st.latex(rf"\angle(p_k - z_j) = \angle({cx_latex(pk)} - ({cx_latex(zj)})) "
                                 rf"= \angle({cx_latex(diff)}) = {a:.2f}^\circ")

                # Somas
                soma_ap = sum(ang_polos)
                soma_az = sum(ang_zeros)
                st.markdown("Somatorios:")
                st.latex(rf"\sum \angle(p_k - p_j) = {soma_ap:.2f}^\circ")
                st.latex(rf"\sum \angle(p_k - z_j) = {soma_az:.2f}^\circ")

                # Resultado
                theta = 180.0 - soma_ap + soma_az
                theta = ((theta + 180) % 360) - 180
                t_show = theta % 360
                c_show = (-theta) % 360
                st.markdown("Resultado:")
                st.latex(rf"\theta_d = 180^\circ - ({soma_ap:.2f}^\circ) + ({soma_az:.2f}^\circ) "
                         rf"= {t_show:.2f}^\circ")
                st.latex(rf"\text{{Conjugado: }} {c_show:.2f}^\circ")
                angulos_partida[pk] = theta

        # --- Angulos de chegada ---
        angulos_chegada = {}
        if zeros_cx:
            st.markdown("### Angulos de chegada (zeros complexos)")
            st.markdown("**Formula:**")
            st.latex(r"\theta_a = 180^\circ - \sum_{j \neq k} \angle(z_k - z_j) "
                     r"+ \sum_j \angle(z_k - p_j)")

            for zk in zeros_cx:
                st.markdown(f"---")
                st.markdown(rf"**Zero $z_k = {cx_latex(zk)}$:**")

                ang_zeros = []
                st.markdown("Angulos dos outros zeros:")
                for zj in zeros:
                    if abs(zj - zk) > 1e-10:
                        a = np.degrees(np.angle(zk - zj))
                        ang_zeros.append(a)
                        st.latex(rf"\angle(z_k - z_j) = {a:.2f}^\circ")

                ang_polos = []
                st.markdown("Angulos dos polos:")
                for pj in polos:
                    a = np.degrees(np.angle(zk - pj))
                    ang_polos.append(a)
                    st.latex(rf"\angle(z_k - p_j) = {a:.2f}^\circ")

                soma_az = sum(ang_zeros)
                soma_ap = sum(ang_polos)
                st.latex(rf"\sum \angle(z_k - z_j) = {soma_az:.2f}^\circ")
                st.latex(rf"\sum \angle(z_k - p_j) = {soma_ap:.2f}^\circ")

                theta = 180.0 - soma_az + soma_ap
                theta = ((theta + 180) % 360) - 180
                t_show = theta % 360
                c_show = (-theta) % 360
                st.latex(rf"\theta_a = 180^\circ - ({soma_az:.2f}^\circ) + ({soma_ap:.2f}^\circ) "
                         rf"= {t_show:.2f}^\circ")
                st.latex(rf"\text{{Conjugado: }} {c_show:.2f}^\circ")
                angulos_chegada[zk] = theta

        # --- Grafico ---
        st.markdown("---")
        fig10, ax10 = plt.subplots(figsize=(10, 7))
        desenhar_lgr_fundo(ax10, todas_raizes, xl, yl)
        desenhar_polos_zeros(ax10, polos, zeros)

        todos_arr = np.concatenate([polos, zeros]) if len(zeros) > 0 else polos
        spread = max(np.ptp(todos_arr.real), np.ptp(todos_arr.imag), 1.0)
        arrow_len = spread * 0.3

        for pk, theta in angulos_partida.items():
            rad = np.radians(theta)
            dx = arrow_len * np.cos(rad)
            dy = arrow_len * np.sin(rad)
            ax10.annotate('', xy=(pk.real + dx, pk.imag + dy),
                          xytext=(pk.real, pk.imag),
                          arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
            ax10.text(pk.real + dx * 1.3, pk.imag + dy * 1.3,
                      f'{theta % 360:.1f}°', color='darkred', fontsize=9, ha='center')
            rad_c = np.radians(-theta)
            dx_c = arrow_len * np.cos(rad_c)
            dy_c = arrow_len * np.sin(rad_c)
            ax10.annotate('', xy=(pk.real + dx_c, -pk.imag + dy_c),
                          xytext=(pk.real, -pk.imag),
                          arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

        for zk, theta in angulos_chegada.items():
            rad = np.radians(theta)
            dx = arrow_len * np.cos(rad)
            dy = arrow_len * np.sin(rad)
            ax10.annotate('', xy=(zk.real + dx, zk.imag + dy),
                          xytext=(zk.real, zk.imag),
                          arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
            ax10.text(zk.real + dx * 1.3, zk.imag + dy * 1.3,
                      f'{theta % 360:.1f}°', color='darkgreen', fontsize=9, ha='center')

        finalizar_grafico(ax10, (xl[0] - marg * 0.5, xl[1] + marg * 0.5),
                          yl + marg * 0.5,
                          "LGR - Angulos de Partida/Chegada", lim_usr)
        fig10.tight_layout()
        st.pyplot(fig10)
        plt.close(fig10)
    else:
        st.info("Sem polos/zeros complexos. Este passo nao se aplica.")

# ============================================================
# Passo 11 - Criterio de angulo
# ============================================================
with st.expander("**Passo 11** - Criterio de angulo"):
    st.markdown("**Condicao de pertinencia ao LGR:**")
    st.latex(r"\sum \angle(s_0 - z_j) - \sum \angle(s_0 - p_i) = \pm 180^\circ (2q+1)")

    st.markdown(rf"**Ponto de teste:** $s_0 = {cx_latex(s_test)}$")

    # --- Angulos dos polos ---
    st.markdown("---")
    st.markdown(r"**Angulos dos polos** ($\theta_i$):")
    theta_poles = []
    for i, p in enumerate(polos):
        diff = s_test - p
        ang_p = np.degrees(np.angle(diff))
        theta_poles.append(ang_p)
        st.latex(rf"\theta_{{{i+1}}} = \angle(s_0 - p_{{{i+1}}}) "
                 rf"= \angle({cx_latex(s_test)} - ({cx_latex(p)})) "
                 rf"= \angle({cx_latex(diff)}) = {ang_p:.2f}^\circ")
    soma_theta = sum(theta_poles)
    st.latex(rf"\sum \theta_i = {soma_theta:.2f}^\circ")

    # --- Angulos dos zeros ---
    if len(zeros) > 0:
        st.markdown(r"**Angulos dos zeros** ($\phi_j$):")
        phi_zeros = []
        for i, z in enumerate(zeros):
            diff = s_test - z
            ang_z = np.degrees(np.angle(diff))
            phi_zeros.append(ang_z)
            st.latex(rf"\phi_{{{i+1}}} = \angle(s_0 - z_{{{i+1}}}) "
                     rf"= \angle({cx_latex(s_test)} - ({cx_latex(z)})) "
                     rf"= \angle({cx_latex(diff)}) = {ang_z:.2f}^\circ")
        soma_phi = sum(phi_zeros)
        st.latex(rf"\sum \phi_j = {soma_phi:.2f}^\circ")
    else:
        soma_phi = 0.0
        st.markdown(r"Sem zeros finitos: $\sum \phi_j = 0^\circ$")

    # --- Resultado ---
    st.markdown("---")
    st.markdown("**Avaliacao:**")
    delta = soma_theta - soma_phi
    st.latex(rf"\Delta\theta = \sum \theta_i - \sum \phi_j = "
             rf"{soma_theta:.2f}^\circ - {soma_phi:.2f}^\circ = {delta:.2f}^\circ")

    ang_norm_display = ang_n % 360
    st.latex(rf"\text{{Angulo normalizado: }} {ang_norm_display:.2f}^\circ")

    if pert:
        st.success(rf"O ponto **pertence** ao LGR "
                   rf"($\Delta\theta = {ang_norm_display:.2f}^\circ \approx 180^\circ$)")
    else:
        st.warning(rf"O ponto **nao pertence** ao LGR "
                   rf"($\Delta\theta = {ang_norm_display:.2f}^\circ \neq 180^\circ$)")

    # --- Grafico ---
    st.markdown("---")
    fig11, ax11 = plt.subplots(figsize=(10, 6))
    desenhar_polos_zeros(ax11, polos, zeros)
    cor = 'limegreen' if pert else 'red'
    marcador = '*' if pert else 'X'
    lbl = rf"$s_0 = {cx_latex(s_test)}$ ({'pertence' if pert else 'nao pertence'})"
    ax11.plot(s_test.real, s_test.imag, marcador, ms=14, color=cor,
              markeredgecolor='black', label=lbl, zorder=6)
    for p in polos:
        ax11.plot([p.real, s_test.real], [p.imag, s_test.imag],
                  ':', color='red', alpha=0.4)
    for z in zeros:
        ax11.plot([z.real, s_test.real], [z.imag, s_test.imag],
                  ':', color='green', alpha=0.4)
    all_x = list(polos.real) + [s_test.real]
    if len(zeros) > 0:
        all_x.extend(list(zeros.real))
    all_y = list(abs(polos.imag)) + [abs(s_test.imag)]
    if len(zeros) > 0:
        all_y.extend(list(abs(zeros.imag)))
    x_lim11 = (min(all_x) - 2, max(all_x) + 2)
    y_lim11 = max(all_y) + 2
    finalizar_grafico(ax11, x_lim11, y_lim11, "Criterio de Angulo", lim_usr)
    fig11.tight_layout()
    st.pyplot(fig11)
    plt.close(fig11)

# ============================================================
# Passo 12 - Calculo de K
# ============================================================
with st.expander("**Passo 12** - Calculo de $K$"):
    st.markdown("**Formula do criterio de modulo:**")
    st.latex(r"K = \frac{\prod_{i} |s_0 - p_i|}{\prod_{j} |s_0 - z_j|}")

    st.markdown(rf"**Ponto:** $s_0 = {cx_latex(s_test)}$")

    # --- Distancias dos polos ---
    st.markdown("---")
    st.markdown("**Distancias dos polos:**")
    prod_p = 1.0
    dist_p_strs = []
    for i, p in enumerate(polos):
        diff = s_test - p
        d = abs(diff)
        prod_p *= d
        dist_p_strs.append(f"{d:.4f}")
        st.latex(rf"|s_0 - p_{{{i+1}}}| = |{cx_latex(s_test)} - ({cx_latex(p)})| "
                 rf"= |{cx_latex(diff)}| = {d:.4f}")

    st.markdown("Produto das distancias dos polos:")
    prod_str = r" \cdot ".join(dist_p_strs)
    st.latex(rf"\prod |s_0 - p_i| = {prod_str} = {prod_p:.4f}")

    # --- Distancias dos zeros ---
    if len(zeros) > 0:
        st.markdown("---")
        st.markdown("**Distancias dos zeros:**")
        prod_z = 1.0
        dist_z_strs = []
        for i, z in enumerate(zeros):
            diff = s_test - z
            d = abs(diff)
            prod_z *= d
            dist_z_strs.append(f"{d:.4f}")
            st.latex(rf"|s_0 - z_{{{i+1}}}| = |{cx_latex(s_test)} - ({cx_latex(z)})| "
                     rf"= |{cx_latex(diff)}| = {d:.4f}")

        st.markdown("Produto das distancias dos zeros:")
        prod_z_str = r" \cdot ".join(dist_z_strs)
        st.latex(rf"\prod |s_0 - z_j| = {prod_z_str} = {prod_z:.4f}")
    else:
        prod_z = 1.0
        st.markdown(r"Sem zeros finitos: $\prod |s_0 - z_j| = 1$")

    # --- K ---
    st.markdown("---")
    st.markdown("**Resultado:**")
    if prod_z > 1e-12:
        K_calc = prod_p / prod_z
        st.latex(rf"K = \frac{{{prod_p:.4f}}}{{{prod_z:.4f}}} = {K_calc:.4f}")
        if pert:
            st.success(rf"O ponto pertence ao LGR. $K = {K_calc:.6f}$")
        else:
            st.warning(rf"O ponto nao pertence ao LGR. "
                       rf"$K = {K_calc:.6f}$ (valor de referencia)")
    else:
        st.error("Nao e possivel calcular $K$: o ponto coincide com um zero.")

# ============================================================
# Grafico Completo do LGR
# ============================================================
st.markdown("---")
st.subheader("Grafico Completo do LGR")

fig_final, ax_final = plt.subplots(figsize=(10, 7))

# Loop limpo, sem a máscara limitadora e usando linha contínua!
for j in range(todas_raizes.shape[1]):
    ramo = todas_raizes[:, j]
    ax_final.plot(ramo.real, ramo.imag, 'b-', linewidth=2.5, alpha=0.8)

desenhar_polos_zeros(ax_final, polos, zeros)

if sigma_a is not None:
    ax_final.plot(sigma_a, 0, "k+", ms=12, mew=2,
                  label=rf"Centroide ($\sigma_a = {sigma_a:.2f}$)")

finalizar_grafico(ax_final, xl, yl, "Lugar Geométrico das Raízes", lim_usr)
fig_final.tight_layout()
st.pyplot(fig_final)
plt.close(fig_final)
