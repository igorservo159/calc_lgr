"""
app pra calcular LGR - DCA3701 Projeto de SisCon
rodar com: streamlit run app_lgr.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="LGR - 12 Passos", layout="wide")


# === utils ===

def formatar_complexo(c):
    r = round(c.real, 4)
    i = round(c.imag, 4)
    if abs(i) < 1e-10:
        return f"{r}"
    s = "+" if i >= 0 else "-"
    return f"{r} {s} {abs(i)}j"


SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

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


# === os 12 passos ===

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


# === calculo do root locus na mao ===

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


def plotar_lgr(num, den, zeros, polos, sigma_a, ang_ass, bk_pts, cruzamentos):
    """monta o grafico do LGR"""
    Ks, todas_raizes = calcular_lgr(num, den)

    # limites baseado nos polos e zeros (nao nas raizes que vao pro infinito)
    todos = np.concatenate([polos, zeros]) if len(zeros) > 0 else polos
    spread = max(np.ptp(todos.real), np.ptp(todos.imag), 1.0)
    marg = max(1.0, 0.5 * spread)
    xl = (todos.real.min() - marg, todos.real.max() + marg)
    yl = max(abs(todos.imag).max() + marg*0.5, marg)

    fig, ax = plt.subplots(figsize=(10, 7))

    # plotar ramos
    xm = (xl[1] - xl[0]) * 0.1
    ym = yl * 0.1
    for j in range(todas_raizes.shape[1]):
        ramo = todas_raizes[:, j]
        mask = ((ramo.real > xl[0] - xm) & (ramo.real < xl[1] + xm) &
                (ramo.imag > -yl - ym) & (ramo.imag < yl + ym))
        ax.plot(ramo[mask].real, ramo[mask].imag, 'b.', markersize=1.5, alpha=0.7)

    # polos e zeros
    ax.plot(polos.real, polos.imag, "rx", ms=10, mew=2, label="Polos", zorder=5)
    if len(zeros) > 0:
        ax.plot(zeros.real, zeros.imag, "go", ms=8, mew=2, fillstyle="none",
                label="Zeros", zorder=5)

    if sigma_a is not None:
        ax.plot(sigma_a, 0, "k+", ms=10, mew=2, label=f"Centroide ({sigma_a:.2f})")

    ax.set_xlim(xl)
    ax.set_ylim(-yl, yl)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginario")
    ax.set_title("Lugar Geometrico das Raizes")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# === teste angulo (passo 11) ===

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


# =============================================
#                INTERFACE
# =============================================

st.title("LGR - 12 Passos")
st.caption("DCA-3701 Projeto de Sistemas de Controle - UFRN")

st.markdown("**G(s) = K . N_G(s) / D_G(s)**")
col1, col2 = st.columns(2)
with col1:
    txt_nG = st.text_input("Numerador G(s)", value="1 0.2", help="Coefs em ordem decrescente de s")
with col2:
    txt_dG = st.text_input("Denominador G(s)", value="1 3.5 6 4 0", help="Coefs em ordem decrescente de s")

st.markdown("**H(s) = N_H(s) / D_H(s)**")
col3, col4 = st.columns(2)
with col3:
    txt_nH = st.text_input("Numerador H(s)", value="1 0", help="Coefs em ordem decrescente de s")
with col4:
    txt_dH = st.text_input("Denominador H(s)", value="1", help="Coefs em ordem decrescente de s")


if st.button("Calcular LGR", type="primary"):
    nG = parse_coefs(txt_nG)
    dG = parse_coefs(txt_dG)
    nH = parse_coefs(txt_nH)
    dH = parse_coefs(txt_dH)

    if any(x is None for x in [nG, dG, nH, dH]):
        st.error("Confere os coeficientes ai, algo ta errado")
        st.stop()

    # passo 1
    num, den = fazer_passo1(nG, dG, nH, dH)

    st.markdown("---")
    st.subheader("Passo 1 - Equacao Caracteristica")
    st.text(f"1 + K[{mostrar_poly(num)}] / [{mostrar_poly(den)}] = 0")
    st.text(f"P(s) = {mostrar_poly(den)} + K({mostrar_poly(num)})")

    # passo 2
    zeros = np.roots(num)
    polos = np.roots(den)

    st.subheader("Passo 2 - Polos e Zeros de MA")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Polos (n_p = {len(polos)})**")
        for p in polos:
            st.text(f"  p = {formatar_complexo(p)}")
    with c2:
        st.markdown(f"**Zeros (n_z = {len(zeros)})**")
        if len(zeros) > 0:
            for z in zeros:
                st.text(f"  z = {formatar_complexo(z)}")
        else:
            st.text("  nenhum")

    # passo 3
    st.subheader("Passo 3 - Polos e Zeros no plano s")
    st.text("(marcados no grafico abaixo)")

    # passo 4
    st.subheader("Passo 4 - Segmentos no eixo real")
    segs = achar_segmentos_eixo_real(zeros, polos)
    if segs:
        for a, b in segs:
            ea = f"{a:.4f}" if np.isfinite(a) else "-inf"
            eb = f"{b:.4f}" if np.isfinite(b) else "+inf"
            st.text(f"  [{ea} , {eb}]")
    else:
        st.text("  Nenhum segmento no eixo real")

    # passo 5
    ls = max(len(polos), len(zeros))
    st.subheader("Passo 5 - Lugares separados")
    st.text(f"  LS = max({len(polos)}, {len(zeros)}) = {ls}")

    # passo 6
    st.subheader("Passo 6 - Simetria")
    st.text("  O LGR e simetrico em relacao ao eixo real")

    # passo 7
    st.subheader("Passo 7 - Assintotas")
    sigma_a, angs = calcular_assintotas(zeros, polos)
    if sigma_a is not None:
        st.text(f"  Centroide: sigma_a = {sigma_a:.4f}")
        st.text(f"  Angulos: {', '.join(f'{a:.1f}\u00b0' for a in angs)}")
    else:
        st.text("  n_p = n_z, sem assintotas")

    # passo 8
    st.subheader("Passo 8 - Breakaway / Break-in")
    bk_pts, bk_info = achar_breakaway(num, den, polos, zeros)
    for txt in bk_info:
        st.text(f"  {txt}")

    # passo 9
    st.subheader("Passo 9 - Cruzamento eixo imaginario")

    # tabela de routh
    routh_linhas, routh_conds, k_crits = tabela_routh(den, num)
    if routh_linhas is not None:
        with st.expander("Tabela de Routh-Hurwitz"):
            for l in routh_linhas:
                st.text(l)
            st.markdown("**Condicoes de estabilidade:**")
            for c in routh_conds:
                st.text(f"  {c}")
            if k_crits:
                st.text(f"  K critico(s): {', '.join(f'{k:.4f}' for k in k_crits)}")

    cruzs, info_jw = cruzamento_jw(den, num)
    st.text(f"  Eq. eliminando K: {mostrar_poly(info_jw['cross'], 'w')} = 0")
    if cruzs:
        for Kc, wc in cruzs:
            st.text(f"  K = {Kc:.4f},  s = +/-{wc:.4f}j")
    else:
        st.text("  Nao cruza o eixo jw (K > 0)")

    # passo 10
    st.subheader("Passo 10 - Angulos de partida/chegada")
    ang_textos = angulos_partida_chegada(zeros, polos)
    if ang_textos:
        for t in ang_textos:
            st.text(f"  {t}")
    else:
        st.text("  Sem polos/zeros complexos")

    # passo 11 e 12 - interativo
    st.subheader("Passo 11/12 - Testar ponto e calcular K")
    cc1, cc2 = st.columns(2)
    with cc1:
        sr = st.number_input("Parte real", value=0.0, format="%.4f", key="sr")
    with cc2:
        si = st.number_input("Parte imaginaria", value=0.0, format="%.4f", key="si")

    s_test = complex(sr, si)
    ang, ang_n, pert, Kp = testar_angulo(s_test, zeros, polos)
    K_ponto = calcular_K_ponto(s_test, zeros, polos)

    if pert:
        st.success(f"Ponto PERTENCE ao LGR (angulo = {ang_n % 360:.2f}\u00b0). K = {Kp:.6f}")
    else:
        st.warning(f"Ponto NAO pertence (angulo = {ang_n % 360:.2f}\u00b0). K nesse ponto = {K_ponto:.6f}")

    # grafico
    st.markdown("---")
    st.subheader("Grafico do LGR")
    fig = plotar_lgr(num, den, zeros, polos, sigma_a, angs, bk_pts, cruzs)
    st.pyplot(fig)
