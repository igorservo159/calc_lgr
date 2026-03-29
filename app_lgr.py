import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="LGR - 12 Passos", layout="wide")

def formatar_complexo(c):
    r = round(c.real, 4)
    i = round(c.imag, 4)
    if abs(i) < 1e-10:
        return f"{r}"
    s = "+" if i >= 0 else "-"
    return f"{r} {s} {abs(i)}j"


SUP = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")

def superscript(n):
    return str(n).translate(SUP)


def mostrar_poly(coefs, var="s"):
    """mostra polinomio bonitinho"""
    grau = len(coefs) - 1
    termos = []
    for k, c in enumerate(coefs):
        exp = grau - k
        if abs(c) < 1e-12:
            continue
        ac = abs(c)

        if exp == 0:
            coef_str = f"{ac:g}"
        elif abs(ac - 1) < 1e-12:
            coef_str = ""
        else:
            coef_str = f"{ac:g}"

        if exp == 0:
            t = coef_str
        elif exp == 1:
            t = f"{coef_str}{var}" if coef_str else var
        else:
            t = f"{coef_str}{var}{superscript(exp)}" if coef_str else f"{var}{superscript(exp)}"

        if not termos:
            termos.append(f"-{t}" if c < 0 else t)
        else:
            termos.append(f"- {t}" if c < 0 else f"+ {t}")

    return " ".join(termos) if termos else "0"


def parse_coefs(texto):
    """pega string de coeficientes e transforma em array"""
    try:
        vals = [float(x) for x in texto.strip().split()]
        if len(vals) == 0:
            return None
        return np.array(vals)
    except:
        return None


def separar_jw(coefs):
    """
    substitui s = jw no polinomio e separa real e imaginario
    j^0=1, j^1=j, j^2=-1, j^3=-j ...
    """
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
    # igualar tamanho
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
        meio = (fronteiras[i] + fronteiras[i+1]) / 2
        cont = sum(1 for r in reais if r > meio + 1e-10)
        if cont % 2 == 1:
            segs.append((fronteiras[i+1], fronteiras[i]))

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
    angs = [(2*q + 1) * 180.0 / diff for q in range(diff)]
    return sigma, angs


def achar_breakaway(num, den, polos, zeros):
    dN = np.polyder(num)
    dD = np.polyder(den)
    eq = np.polysub(np.convolve(num, dD), np.convolve(den, dN))
    raizes = np.roots(eq)

    reais_pz = []
    for p in polos:
        if abs(p.imag) < 1e-8: reais_pz.append(p.real)
    for z in zeros:
        if abs(z.imag) < 1e-8: reais_pz.append(z.real)

    pts = []
    info = []
    for r in raizes:
        vn = np.polyval(num, r)
        Kv = -np.polyval(den, r) / vn if abs(vn) > 1e-12 else None

        if abs(r.imag) < 1e-6:
            rr = r.real
            cont = sum(1 for x in reais_pz if x > rr + 1e-10)
            no_lgr = cont % 2 == 1
            Kr = Kv.real if Kv is not None else np.inf
            status = "no LGR" if no_lgr else "fora do LGR"
            info.append(f"s = {rr:.4f} (K = {Kr:.4f}) [{status}]")
            if no_lgr and Kr > 0:
                pts.append((rr, Kr))
        else:
            if Kv is not None and abs(Kv.imag) < 1e-6 and Kv.real > 0:
                info.append(f"s = {formatar_complexo(r)} (K = {Kv.real:.4f}) [no LGR]")
                pts.append((complex(r), Kv.real))
            else:
                Ks = formatar_complexo(Kv) if Kv is not None else "inf"
                info.append(f"s = {formatar_complexo(r)} (K = {Ks}) [fora]")

    return pts, info


def cruzamento_jw(den, num):
    """acha onde o LGR cruza o eixo imaginario"""
    Re_D, Im_D = separar_jw(den)
    Re_N, Im_N = separar_jw(num)

    # eliminando K: Re_D*Im_N - Im_D*Re_N = 0
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
            if not any(abs(K - kk) < 1e-4 and abs(omega - ww) < 1e-4 for kk, ww in resultado):
                resultado.append((K, omega))

    return resultado, info_jw


def angulos_partida_chegada(zeros, polos):
    textos = []
    polos_cx = [p for p in polos if p.imag > 1e-8]
    zeros_cx = [z for z in zeros if z.imag > 1e-8]

    for pk in polos_cx:
        ap = sum(np.degrees(np.angle(pk - pj)) for pj in polos if abs(pj - pk) > 1e-10)
        az = sum(np.degrees(np.angle(pk - zj)) for zj in zeros)
        theta = 180.0 - ap + az
        theta = ((theta + 180) % 360) - 180
        t_show = theta % 360
        c_show = (-theta) % 360
        textos.append(f"Polo {formatar_complexo(pk)}: partida = {t_show:.2f}\u00b0 (conj: {c_show:.2f}\u00b0)")

    for zk in zeros_cx:
        az = sum(np.degrees(np.angle(zk - zj)) for zj in zeros if abs(zj - zk) > 1e-10)
        ap = sum(np.degrees(np.angle(zk - pj)) for pj in polos)
        theta = 180.0 - az + ap
        theta = ((theta + 180) % 360) - 180
        t_show = theta % 360
        c_show = (-theta) % 360
        textos.append(f"Zero {formatar_complexo(zk)}: chegada = {t_show:.2f}\u00b0 (conj: {c_show:.2f}\u00b0)")

    return textos


def tabela_routh(den, num):
    """tenta montar a tabela de routh (precisa do sympy)"""
    try:
        import sympy as sp
    except ImportError:
        return None, None, None

    K = sp.Symbol('K', positive=True)
    n = max(len(den), len(num))
    dp = np.pad(den, (n - len(den), 0))
    np_ = np.pad(num, (n - len(num), 0))

    def sym(c):
        r = round(c)
        return sp.Integer(r) if abs(c - r) < 1e-10 else sp.nsimplify(c, rational=True)

    coefs = [sym(d) + K * sym(nn) for d, nn in zip(dp, np_)]
    grau = len(coefs) - 1
    cols = (grau + 2) // 2

    tab = [[sp.S.Zero]*cols for _ in range(grau + 1)]
    for j in range(cols):
        if 2*j < len(coefs): tab[0][j] = coefs[2*j]
        if 2*j+1 < len(coefs): tab[1][j] = coefs[2*j+1]

    for i in range(2, grau + 1):
        piv = tab[i-1][0]
        if piv == 0:
            break
        for j in range(cols - 1):
            n_ = tab[i-1][0]*tab[i-2][j+1] - tab[i-2][0]*tab[i-1][j+1]
            tab[i][j] = sp.simplify(n_ / piv)

    # condições de estabilidade
    conds = []
    k_crit = set()
    for i in range(grau + 1):
        e = sp.simplify(tab[i][0])
        if e.has(K):
            try:
                c = sp.solve(e > 0, K)
                if c is True or c == sp.S.true:
                    conds.append(f"s{superscript(grau-i)}: {e} > 0  ->  sempre (K>0)")
                else:
                    conds.append(f"s{superscript(grau-i)}: {e} > 0  ->  {c}")
            except:
                conds.append(f"s{superscript(grau-i)}: {e} > 0")

            sols = sp.solve(sp.Eq(e, 0), K)
            for s in sols:
                if s.is_real and s > 0:
                    k_crit.add(float(s))

    # formatar tabela
    linhas = []
    for i in range(grau + 1):
        exp = grau - i
        vals = [str(sp.simplify(tab[i][j])) for j in range(cols) if tab[i][j] != 0 or j == 0]
        linhas.append(f"s{superscript(exp)} | " + "  ".join(vals))

    return linhas, conds, sorted(k_crit)


def ordenar_raizes(prev, curr):
    """ordena as raizes pra manter continuidade entre passos de K"""
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
    """varre K de 0 ate Kmax e acha as raizes do polinomio caracteristico"""
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

    # mistura linear (perto de 0) com log (longe)
    k1 = np.linspace(0, 0.1, 200)
    k2 = np.logspace(-1, np.log10(Kmax), 4800)
    Ks = np.unique(np.concatenate([k1, k2]))
    Ks.sort()

    raizes = np.zeros((len(Ks), np_), dtype=complex)
    for i, k in enumerate(Ks):
        poly = np.polyadd(den, k * num)
        r = np.roots(poly)
        if i > 0:
            r = ordenar_raizes(raizes[i-1], r)
        raizes[i] = r

    return Ks, raizes


def testar_angulo(s_teste, zeros, polos):
    ap = sum(np.degrees(np.angle(s_teste - p)) for p in polos)
    az = sum(np.degrees(np.angle(s_teste - z)) for z in zeros)
    ang = az - ap
    ang_norm = ((ang + 180) % 360) - 180

    pertence = abs(abs(ang_norm) - 180) < 1.0
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


def mostrar_fatorado(raizes, var="s"):
    """mostra polinomio na forma fatorada, agrupando raizes repetidas"""
    if len(raizes) == 0:
        return "1"
    # monta texto de cada fator individual
    fatores_txt = []
    for r in raizes:
        if abs(r.imag) < 1e-8:
            rv = round(r.real, 4)
            if abs(rv) < 1e-8:
                fatores_txt.append(var)
            elif rv > 0:
                fatores_txt.append(f"({var} - {rv:g})")
            else:
                fatores_txt.append(f"({var} + {abs(rv):g})")
        else:
            fatores_txt.append(f"({var} - ({formatar_complexo(r)}))")
    # conta repeticoes mantendo ordem de aparicao
    vistos = []
    contagem = {}
    for f in fatores_txt:
        if f not in contagem:
            contagem[f] = 0
            vistos.append(f)
        contagem[f] += 1
    # monta resultado com expoente quando necessario
    partes = []
    for f in vistos:
        n = contagem[f]
        if n == 1:
            partes.append(f)
        else:
            partes.append(f"{f}{superscript(n)}")
    return "".join(partes)


def limites_grafico(polos, zeros, extra_real=None):
    """calcula limites x,y para graficos de polos/zeros"""
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
    """desenha polos e zeros no grafico"""
    ax.plot(polos.real, polos.imag, "rx", ms=10, mew=2, label="Polos", zorder=5)
    if len(zeros) > 0:
        ax.plot(zeros.real, zeros.imag, "go", ms=8, mew=2, fillstyle="none",
                label="Zeros", zorder=5)


def finalizar_grafico(ax, xl, yl, titulo=""):
    """configura eixos, grid, titulo"""
    ax.set_xlim(xl)
    ax.set_ylim(-yl, yl)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginario")
    ax.set_title(titulo)
    ax.legend(fontsize=9)


def desenhar_segmentos(ax, segs, xl):
    """desenha segmentos do eixo real que pertencem ao LGR"""
    for i, (a, b) in enumerate(segs):
        a_plot = max(a, xl[0] - 5) if np.isfinite(a) else xl[0] - 5
        b_plot = min(b, xl[1] + 5) if np.isfinite(b) else xl[1] + 5
        lbl = "Segmento LGR" if i == 0 else ""
        ax.plot([a_plot, b_plot], [0, 0], 'b-', linewidth=4, alpha=0.6,
                solid_capstyle='round', label=lbl)


def desenhar_assintotas(ax, sigma_a, angs, xl, yl):
    """desenha linhas de assintotas a partir do centroide"""
    line_len = max(abs(xl[0]), abs(xl[1]), yl) * 2
    for i, ang_deg in enumerate(angs):
        ang_rad = np.radians(ang_deg)
        dx = line_len * np.cos(ang_rad)
        dy = line_len * np.sin(ang_rad)
        lbl = "Assintotas" if i == 0 else ""
        ax.plot([sigma_a, sigma_a + dx], [0, dy], '--', color='darkorange',
                linewidth=1.5, alpha=0.7, label=lbl)


def desenhar_lgr_fundo(ax, todas_raizes, xl, yl):
    """desenha o LGR numerico como fundo cinza"""
    xm = (xl[1] - xl[0]) * 0.1
    ym = yl * 0.1
    for j in range(todas_raizes.shape[1]):
        ramo = todas_raizes[:, j]
        mask = ((ramo.real > xl[0] - xm) & (ramo.real < xl[1] + xm) &
                (ramo.imag > -yl - ym) & (ramo.imag < yl + ym))
        ax.plot(ramo[mask].real, ramo[mask].imag, '.', color='gray',
                markersize=1.5, alpha=0.35)


# ============================================================
# Interface
# ============================================================

st.title("LGR - Jo\u00e3o Igor Ramos de Lima")
st.caption("DCA-3701 Projeto de Sistemas de Controle - UFRN")

st.markdown("**G(s) = K . N(G(s)) / D(G(s))**")
col1, col2 = st.columns(2)
with col1:
    txt_nG = st.text_input("Numerador G(s)", value="1 2", help="Coefs em ordem decrescente de s")
with col2:
    txt_dG = st.text_input("Denominador G(s)", value="1 4 0", help="Coefs em ordem decrescente de s")

st.markdown("**H(s) = N_H(s) / D_H(s)**")
col3, col4 = st.columns(2)
with col3:
    txt_nH = st.text_input("Numerador H(s)", value="1", help="Coefs em ordem decrescente de s")
with col4:
    txt_dH = st.text_input("Denominador H(s)", value="1 1", help="Coefs em ordem decrescente de s")

st.markdown("**Ponto de teste (Passos 11/12):**")
ct1, ct2 = st.columns(2)
with ct1:
    sr = st.number_input("Parte real", value=0.0, format="%.4f", key="sr")
with ct2:
    si = st.number_input("Parte imaginaria", value=0.0, format="%.4f", key="si")

if st.button("Calcular LGR", type="primary"):
    st.session_state['calcular_lgr'] = True

if not st.session_state.get('calcular_lgr', False):
    st.stop()

nG = parse_coefs(txt_nG)
dG = parse_coefs(txt_dG)
nH = parse_coefs(txt_nH)
dH = parse_coefs(txt_dH)

if any(x is None for x in [nG, dG, nH, dH]):
    st.error("Confere os coeficientes ai, algo ta errado")
    st.stop()

# ============================================================
# Computacoes (toda logica inalterada)
# ============================================================

num, den = fazer_passo1(nG, dG, nH, dH)
zeros = np.roots(num)
polos = np.roots(den)
segs = achar_segmentos_eixo_real(zeros, polos)
ls = max(len(polos), len(zeros))
sigma_a, angs = calcular_assintotas(zeros, polos)
bk_pts, bk_info = achar_breakaway(num, den, polos, zeros)
routh_linhas, routh_conds, k_crits = tabela_routh(den, num)
cruzs, info_jw = cruzamento_jw(den, num)
ang_textos = angulos_partida_chegada(zeros, polos)
Ks_lgr, todas_raizes = calcular_lgr(num, den)

xl, yl, marg = limites_grafico(polos, zeros,
    [sigma_a] if sigma_a is not None else None)

s_test = complex(sr, si)
ang, ang_n, pert, Kp = testar_angulo(s_test, zeros, polos)
K_ponto = calcular_K_ponto(s_test, zeros, polos)

st.markdown("---")

# ============================================================
# Passo 1
# ============================================================
with st.expander("Passo 1 - Equacao Caracteristica", expanded=True):
    st.text(f"G(s) = K . ({mostrar_poly(nG)}) / ({mostrar_poly(dG)})")
    st.text(f"H(s) = ({mostrar_poly(nH)}) / ({mostrar_poly(dH)})")
    st.text("")
    st.text(f"1 + G(s)H(s) = 1 + K . [{mostrar_poly(num)}] / [{mostrar_poly(den)}]")
    st.text(f"             = 1 + K . P(s) = 0")
    st.text("")
    st.text(f"Eq. Caracteristica: {mostrar_poly(den)} + K({mostrar_poly(num)}) = 0")

# ============================================================
# Passo 2
# ============================================================
with st.expander("Passo 2 - P(s) na forma fatorada"):
    num_fat = mostrar_fatorado(zeros)
    den_fat = mostrar_fatorado(polos)
    st.text(f"1 + G(s)H(s) = 1 + K . {num_fat} / [{den_fat}] = 0")

# ============================================================
# Passo 3
# ============================================================
with st.expander("Passo 3 - Polos e Zeros no plano s"):
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    desenhar_polos_zeros(ax3, polos, zeros)
    for i, p in enumerate(polos):
        ax3.annotate(f"  p{i+1}", (p.real, p.imag), fontsize=9, color='red')
    for i, z in enumerate(zeros):
        ax3.annotate(f"  z{i+1}", (z.real, z.imag), fontsize=9, color='green')
    finalizar_grafico(ax3, xl, yl, "Polos e Zeros no plano s")
    fig3.tight_layout()
    st.pyplot(fig3)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Polos (n_p = {len(polos)})**")
        for i, p in enumerate(polos):
            st.text(f"  p{i+1} = {formatar_complexo(p)}")
    with c2:
        st.markdown(f"**Zeros (n_z = {len(zeros)})**")
        if len(zeros) > 0:
            for i, z in enumerate(zeros):
                st.text(f"  z{i+1} = {formatar_complexo(z)}")
        else:
            st.text("  nenhum")

# ============================================================
# Passo 4
# ============================================================
with st.expander("Passo 4 - Segmentos no eixo real"):
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    desenhar_polos_zeros(ax4, polos, zeros)
    desenhar_segmentos(ax4, segs, xl)
    finalizar_grafico(ax4, xl, yl, "Segmentos do eixo real pertencentes ao LGR")
    fig4.tight_layout()
    st.pyplot(fig4)

    st.markdown("**Regra:** segmentos a esquerda de um numero impar de polos e zeros reais")
    if segs:
        for a, b in segs:
            ea = f"{a:.4f}" if np.isfinite(a) else "-inf"
            eb = f"{b:.4f}" if np.isfinite(b) else "+inf"
            st.text(f"  [{ea} , {eb}]")
    else:
        st.text("  Nenhum segmento no eixo real")

# ============================================================
# Passo 5
# ============================================================
with st.expander("Passo 5 - Lugares separados"):
    st.text(f"  n_p = {len(polos)},  n_z = {len(zeros)}")
    st.text(f"  LS = max(n_p, n_z) = max({len(polos)}, {len(zeros)}) = {ls}")

# ============================================================
# Passo 6
# ============================================================
with st.expander("Passo 6 - Simetria"):
    st.text("  O LGR e simetrico em relacao ao eixo real")
    st.text("  (raizes complexas sempre ocorrem em pares conjugados)")

# ============================================================
# Passo 7
# ============================================================
with st.expander("Passo 7 - Assintotas"):
    if sigma_a is not None:
        na = len(polos) - len(zeros)
        st.markdown(f"**Numero de assintotas:** n_a = n_p - n_z = {len(polos)} - {len(zeros)} = {na}")

        soma_p = np.sum(polos).real
        soma_z = np.sum(zeros).real if len(zeros) > 0 else 0.0
        st.markdown("**Centroide (ponto de encontro das assintotas):**")
        st.text(f"  sigma_a = (soma polos - soma zeros) / (n_p - n_z)")
        st.text(f"  sigma_a = ({soma_p:.4f} - {soma_z:.4f}) / {na}")
        st.text(f"  sigma_a = {sigma_a:.4f}")

        st.markdown("**Angulos das assintotas:**")
        st.text(f"  phi_a = (2q + 1) . 180 / {na}")
        for q, a in enumerate(angs):
            st.text(f"  q = {q}: phi_a = {a:.1f}\u00b0")

        fig7, ax7 = plt.subplots(figsize=(10, 6))
        desenhar_polos_zeros(ax7, polos, zeros)
        desenhar_segmentos(ax7, segs, xl)
        desenhar_assintotas(ax7, sigma_a, angs, xl, yl)
        ax7.plot(sigma_a, 0, "k+", ms=12, mew=2, label=f"Centroide ({sigma_a:.2f})")
        finalizar_grafico(ax7, xl, yl, "LGR - Assintotas")
        fig7.tight_layout()
        st.pyplot(fig7)
    else:
        st.text("  n_p = n_z, sem assintotas")

# ============================================================
# Passo 8
# ============================================================
with st.expander("Passo 8 - Breakaway / Break-in"):
    st.markdown("**Pontos de saida/entrada (descolamento):**")
    st.text("  K = -D(s)/N(s),  derivar e igualar a zero")
    st.text("  N'(s)D(s) - N(s)D'(s) = 0")
    st.text("")
    st.markdown("**Raizes encontradas:**")
    for txt in bk_info:
        st.text(f"  {txt}")

    fig8, ax8 = plt.subplots(figsize=(10, 6))
    desenhar_polos_zeros(ax8, polos, zeros)
    desenhar_segmentos(ax8, segs, xl)
    if sigma_a is not None:
        desenhar_assintotas(ax8, sigma_a, angs, xl, yl)
    bk_validos = [(s_bk, k_bk) for s_bk, k_bk in bk_pts if isinstance(s_bk, (int, float))]
    if bk_validos:
        bk_x = [s_bk for s_bk, _ in bk_validos]
        ax8.plot(bk_x, [0]*len(bk_x), 'md', ms=10, mew=2,
                label="Breakaway/Break-in", zorder=6)
        for s_bk, k_bk in bk_validos:
            ax8.annotate(f"  s={s_bk:.2f}, K={k_bk:.2f}", (s_bk, 0),
                        fontsize=8, color='purple')
    finalizar_grafico(ax8, xl, yl, "LGR - Pontos de Descolamento")
    fig8.tight_layout()
    st.pyplot(fig8)

# ============================================================
# Passo 9
# ============================================================
with st.expander("Passo 9 - Cruzamento eixo imaginario"):
    # tabela de Routh
    if routh_linhas is not None:
        st.markdown("**Tabela de Routh-Hurwitz:**")
        st.code("\n".join(routh_linhas), language=None)
        st.markdown("**Condicoes de estabilidade:**")
        for c in routh_conds:
            st.text(f"  {c}")
        if k_crits:
            st.text(f"  K critico(s): {', '.join(f'{k:.4f}' for k in k_crits)}")

    st.markdown("**Cruzamento com eixo imaginario:**")
    st.text(f"  Substituindo s = jw e eliminando K:")
    st.text(f"  {mostrar_poly(info_jw['cross'], 'w')} = 0")
    if cruzs:
        for Kc, wc in cruzs:
            st.text(f"  K = {Kc:.4f},  s = +/- {wc:.4f}j")
    else:
        st.text("  Nao cruza o eixo jw (K > 0)")

    fig9, ax9 = plt.subplots(figsize=(10, 6))
    desenhar_lgr_fundo(ax9, todas_raizes, xl, yl)
    desenhar_polos_zeros(ax9, polos, zeros)
    if cruzs:
        for Kc, wc in cruzs:
            ax9.plot(0, wc, 's', ms=10, color='cyan', markeredgecolor='navy',
                    mew=2, zorder=6, label=f"jw = {wc:.2f}j (K={Kc:.2f})")
            ax9.plot(0, -wc, 's', ms=10, color='cyan', markeredgecolor='navy',
                    mew=2, zorder=6)
    finalizar_grafico(ax9, xl, yl, "LGR - Cruzamento com eixo imaginario")
    fig9.tight_layout()
    st.pyplot(fig9)

# ============================================================
# Passo 10
# ============================================================
with st.expander("Passo 10 - Angulos de partida/chegada"):
    polos_cx = [p for p in polos if p.imag > 1e-8]
    zeros_cx = [z for z in zeros if z.imag > 1e-8]

    if polos_cx or zeros_cx:
        # calculos detalhados de partida
        angulos_partida = {}
        for pk in polos_cx:
            st.markdown(f"**Polo {formatar_complexo(pk)} - angulo de partida:**")
            ang_polos_list = []
            for pj in polos:
                if abs(pj - pk) > 1e-10:
                    a_pj = np.degrees(np.angle(pk - pj))
                    ang_polos_list.append(a_pj)
                    st.text(f"  angulo do polo {formatar_complexo(pj)}: {a_pj:.2f}\u00b0")
            ang_zeros_list = []
            for zj in zeros:
                a_zj = np.degrees(np.angle(pk - zj))
                ang_zeros_list.append(a_zj)
                st.text(f"  angulo do zero {formatar_complexo(zj)}: {a_zj:.2f}\u00b0")
            soma_ap = sum(ang_polos_list)
            soma_az = sum(ang_zeros_list)
            theta = 180.0 - soma_ap + soma_az
            theta = ((theta + 180) % 360) - 180
            st.text(f"  partida = 180 - {soma_ap:.2f} + {soma_az:.2f} = {theta % 360:.2f}\u00b0")
            angulos_partida[pk] = theta

        # calculos detalhados de chegada
        angulos_chegada = {}
        for zk in zeros_cx:
            st.markdown(f"**Zero {formatar_complexo(zk)} - angulo de chegada:**")
            ang_zeros_list = []
            for zj in zeros:
                if abs(zj - zk) > 1e-10:
                    a_zj = np.degrees(np.angle(zk - zj))
                    ang_zeros_list.append(a_zj)
                    st.text(f"  angulo do zero {formatar_complexo(zj)}: {a_zj:.2f}\u00b0")
            ang_polos_list = []
            for pj in polos:
                a_pj = np.degrees(np.angle(zk - pj))
                ang_polos_list.append(a_pj)
                st.text(f"  angulo do polo {formatar_complexo(pj)}: {a_pj:.2f}\u00b0")
            soma_az = sum(ang_zeros_list)
            soma_ap = sum(ang_polos_list)
            theta = 180.0 - soma_az + soma_ap
            theta = ((theta + 180) % 360) - 180
            st.text(f"  chegada = 180 - {soma_az:.2f} + {soma_ap:.2f} = {theta % 360:.2f}\u00b0")
            angulos_chegada[zk] = theta

        st.markdown("**Resumo:**")
        for t in ang_textos:
            st.text(f"  {t}")

        # grafico
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
            ax10.text(pk.real + dx*1.3, pk.imag + dy*1.3,
                     f'{theta%360:.1f}\u00b0', color='darkred', fontsize=9, ha='center')
            # conjugado
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
            ax10.text(zk.real + dx*1.3, zk.imag + dy*1.3,
                     f'{theta%360:.1f}\u00b0', color='darkgreen', fontsize=9, ha='center')

        finalizar_grafico(ax10, (xl[0] - marg*0.5, xl[1] + marg*0.5),
                         yl + marg*0.5,
                         "LGR - Angulos de Partida/Chegada")
        fig10.tight_layout()
        st.pyplot(fig10)
    else:
        st.text("  Sem polos/zeros complexos, passo nao se aplica")

# ============================================================
# Passo 11
# ============================================================
with st.expander("Passo 11 - Criterio de Angulo"):
    st.markdown(f"**Ponto de teste:** s = {formatar_complexo(s_test)}")

    st.markdown("**Angulos dos polos:**")
    theta_poles = []
    for i, p in enumerate(polos):
        ang_p = np.degrees(np.angle(s_test - p))
        theta_poles.append(ang_p)
        st.text(f"  polo p{i+1} = {formatar_complexo(p)}: theta = {ang_p:.2f}\u00b0")
    soma_theta = sum(theta_poles)
    st.text(f"  Soma angulos polos = {soma_theta:.2f}\u00b0")

    if len(zeros) > 0:
        st.markdown("**Angulos dos zeros:**")
        phi_zeros = []
        for i, z in enumerate(zeros):
            ang_z = np.degrees(np.angle(s_test - z))
            phi_zeros.append(ang_z)
            st.text(f"  zero z{i+1} = {formatar_complexo(z)}: phi = {ang_z:.2f}\u00b0")
        soma_phi = sum(phi_zeros)
        st.text(f"  Soma angulos zeros = {soma_phi:.2f}\u00b0")
    else:
        soma_phi = 0.0
        st.text("  Nenhum zero (soma phi = 0)")

    st.markdown("**Resultado:**")
    st.text(f"  Angulo = soma_polos - soma_zeros = {soma_theta:.2f} - {soma_phi:.2f} = {soma_theta - soma_phi:.2f}\u00b0")
    st.text(f"  Angulo normalizado: {ang_n % 360:.2f}\u00b0")

    if pert:
        st.success(f"Ponto PERTENCE ao LGR (angulo = {ang_n % 360:.2f}\u00b0)")
    else:
        st.warning(f"Ponto NAO pertence ao LGR (angulo = {ang_n % 360:.2f}\u00b0)")

    # grafico
    fig11, ax11 = plt.subplots(figsize=(10, 6))
    desenhar_polos_zeros(ax11, polos, zeros)
    cor = 'limegreen' if pert else 'red'
    marcador = '*' if pert else 'X'
    lbl = f"s = {formatar_complexo(s_test)} ({'pertence' if pert else 'nao pertence'})"
    ax11.plot(s_test.real, s_test.imag, marcador, ms=14, color=cor,
             markeredgecolor='black', label=lbl, zorder=6)
    for p in polos:
        ax11.plot([p.real, s_test.real], [p.imag, s_test.imag], ':', color='red', alpha=0.4)
    for z in zeros:
        ax11.plot([z.real, s_test.real], [z.imag, s_test.imag], ':', color='green', alpha=0.4)
    all_x = list(polos.real) + [s_test.real]
    if len(zeros) > 0:
        all_x.extend(list(zeros.real))
    all_y = list(abs(polos.imag)) + [abs(s_test.imag)]
    if len(zeros) > 0:
        all_y.extend(list(abs(zeros.imag)))
    x_lim11 = (min(all_x) - 2, max(all_x) + 2)
    y_lim11 = max(all_y) + 2
    finalizar_grafico(ax11, x_lim11, y_lim11, "Criterio de Angulo")
    fig11.tight_layout()
    st.pyplot(fig11)

# ============================================================
# Passo 12
# ============================================================
with st.expander("Passo 12 - Calculo de K"):
    st.markdown(f"**Ponto:** s = {formatar_complexo(s_test)}")

    st.markdown("**Distancias dos polos:**")
    prod_p = 1.0
    for i, p in enumerate(polos):
        d = abs(s_test - p)
        prod_p *= d
        st.text(f"  |s - p{i+1}| = {d:.4f}")
    st.text(f"  Produto distancias polos = {prod_p:.4f}")

    if len(zeros) > 0:
        st.markdown("**Distancias dos zeros:**")
        prod_z = 1.0
        for i, z in enumerate(zeros):
            d = abs(s_test - z)
            prod_z *= d
            st.text(f"  |s - z{i+1}| = {d:.4f}")
        st.text(f"  Produto distancias zeros = {prod_z:.4f}")
    else:
        prod_z = 1.0
        st.text("  Nenhum zero (produto = 1)")

    if prod_z > 1e-12:
        K_calc = prod_p / prod_z
        st.text(f"  K = {prod_p:.4f} / {prod_z:.4f} = {K_calc:.4f}")
        if pert:
            st.success(f"Ponto pertence ao LGR. K = {K_calc:.6f}")
        else:
            st.warning(f"Ponto nao pertence ao LGR. K = {K_calc:.6f} (referencia apenas)")
    else:
        st.text("  Nao e possivel calcular K (ponto coincide com zero)")

# ============================================================
# Grafico Completo
# ============================================================
st.markdown("---")
st.subheader("Grafico Completo do LGR")

fig_final, ax_final = plt.subplots(figsize=(10, 7))
xm = (xl[1] - xl[0]) * 0.1
ym = yl * 0.1
for j in range(todas_raizes.shape[1]):
    ramo = todas_raizes[:, j]
    mask = ((ramo.real > xl[0] - xm) & (ramo.real < xl[1] + xm) &
            (ramo.imag > -yl - ym) & (ramo.imag < yl + ym))
    ax_final.plot(ramo[mask].real, ramo[mask].imag, 'b.', markersize=1.5, alpha=0.7)

desenhar_polos_zeros(ax_final, polos, zeros)
if sigma_a is not None:
    ax_final.plot(sigma_a, 0, "k+", ms=10, mew=2, label=f"Centroide ({sigma_a:.2f})")
finalizar_grafico(ax_final, xl, yl, "Lugar Geometrico das Raizes")
fig_final.tight_layout()
st.pyplot(fig_final)
