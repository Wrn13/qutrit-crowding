"""Poster figure: in-phase (I) and quadrature (Q) components of a DRAG pulse.

The in-phase drive is the raised-cosine envelope used in the SNAIL iSWAP suite,
    Omega_I(t) = (A/2) [1 - cos(2 pi t / t_g)],
and the first-order DRAG quadrature is its scaled negative time-derivative
(Motzoi et al., PRL 103, 110501 (2009)),
    Omega_Q(t) = -d/dt Omega_I(t) / (2 pi delta),
with delta the detuning of the off-resonant transition being suppressed. The
quadrature amplitude scales as 1/delta; a representative delta is used here so
both traces are legible on one axis.
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# --- colours --------------------------------------------------------------
RED = "#C0392B"
BLUE = "#2980B9"

# --- pulse definition ------------------------------------------------------
t_g = 92.0                                   # gate duration (ns)
delta_GHz = 0.013                            # representative DRAG detuning (GHz)
t = np.linspace(0.0, t_g, 1000)

Omega_I = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / t_g))          # raised cosine, peak 1
dOmega_I = (np.pi / t_g) * np.sin(2.0 * np.pi * t / t_g)        # d/dt Omega_I
Omega_Q = -dOmega_I / (2.0 * np.pi * delta_GHz)                # DRAG quadrature

# --- figure ----------------------------------------------------------------
plt.rcParams.update({
    "font.size": 15,
    "font.family": "sans-serif",
    "axes.linewidth": 1.4,
    "mathtext.fontset": "dejavusans",
})
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

ax.axhline(0.0, color="0.75", lw=1.0, zorder=0)
ax.plot(t, Omega_I, color=BLUE, lw=3.2, zorder=3, label=r"in-phase  $\Omega_I$")
ax.plot(t, Omega_Q, color=RED, lw=3.2, zorder=3, label=r"quadrature  $\Omega_Q$")

ax.set_xlabel("Time  (ns)", fontsize=16)
ax.set_ylabel("Drive Amplitude", fontsize=16)
ax.set_xlim(0, t_g)
ax.set_ylim(-0.7, 1.12)
ax.set_title("Pulse Shape with DRAG",
             fontsize=16, color=BLUE, pad=12, fontweight="bold")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(width=1.4, labelsize=13)

leg = ax.legend(loc="upper right", frameon=False, fontsize=11,
                handlelength=1.4, borderaxespad=0.4)
for txt, col in zip(leg.get_texts(), (BLUE, RED)):
    txt.set_color(col)

fig.tight_layout()
fig.savefig("drag_iq_figure.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig("drag_iq_figure.pdf",
            bbox_inches="tight", facecolor="white")
print("saved drag_iq_figure.png / .pdf ; Q/I peak ratio =",
      round(np.max(np.abs(Omega_Q)) / np.max(Omega_I), 3),
      f"(delta = {delta_GHz*1e3:.0f} MHz)")

"""Poster figure: DRAG reshapes the pulse spectrum to null the spectator.

The complex drive envelope Omega(t) = Omega_I(t) + i Omega_Q(t) with the
first-order DRAG quadrature Omega_Q = -dOmega_I/dt / (2 pi delta_0) has Fourier
transform
    Omega~(nu) = Omega_I~(nu) * (1 + nu / delta_0),
so its spectral weight vanishes at nu = -delta_0. Choosing delta_0 equal to the
spectator beat places that null on the off-resonant spectator transition, which
is why DRAG suppresses spectator driving while leaving the carrier (nu = 0)
untouched. Computed here directly by FFT of the raised-cosine pulse.
"""

RED = "#C0392B"
BLUE = "#2980B9"

# --- pulse + spectator ------------------------------------------------------
t_g = 92.0                        # gate duration (ns)
beat_MHz = -15.0                  # spectator detuning from the pump (MHz); ~1.4/t_g,
                                  #   i.e. on the pulse's spectral shoulder where the
                                  #   drive still has appreciable weight (a 92 ns pulse
                                  #   has bandwidth ~1/t_g ~ 11 MHz, so a far spectator is
                                  #   already bandwidth-suppressed with or without DRAG)
delta0_GHz = -beat_MHz / 1e3      # DRAG coeff placing the null at nu = beat

# zero-padded time grid for fine frequency resolution
dt = 0.5                          # ns  -> Nyquist 1 GHz
T = 2000.0                        # ns  -> df = 0.5 MHz
N = int(T / dt)
t = np.arange(N) * dt
t0 = (T - t_g) / 2.0              # center the pulse in the window
on = (t >= t0) & (t <= t0 + t_g)
tau = t[on] - t0

Omega_I = np.zeros(N)
Omega_Q = np.zeros(N)
Omega_I[on] = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau / t_g))
dI = (np.pi / t_g) * np.sin(2.0 * np.pi * tau / t_g)
Omega_Q[on] = -dI / (2.0 * np.pi * delta0_GHz)

# --- spectra (FFT of real I vs complex I + iQ) ------------------------------
nu = np.fft.fftshift(np.fft.fftfreq(N, dt)) * 1e3        # MHz
S_nodrag = np.fft.fftshift(np.fft.fft(Omega_I))
S_drag = np.fft.fftshift(np.fft.fft(Omega_I + 1j * Omega_Q))
norm = np.max(np.abs(S_nodrag))
w_nodrag = np.abs(S_nodrag) / norm
w_drag = np.abs(S_drag) / norm

# value at the spectator, for the marker + a sanity print
k = int(np.argmin(np.abs(nu - beat_MHz)))
print(f"weight at spectator (nu={beat_MHz:.0f} MHz):  "
      f"no-DRAG = {w_nodrag[k]:.3f},  DRAG = {w_drag[k]:.4f}")

# --- figure -----------------------------------------------------------------
plt.rcParams.update({
    "font.size": 15, "font.family": "sans-serif",
    "axes.linewidth": 1.4, "mathtext.fontset": "dejavusans",
})
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

ax.axvline(beat_MHz, color="0.6", lw=1.2, ls=":", zorder=1)
ax.plot(nu, w_nodrag, color=BLUE, lw=3.2, zorder=3, label="no DRAG")
ax.plot(nu, w_drag, color=RED, lw=3.2, zorder=3, label="with DRAG")

# emphasize the suppression at the spectator
ax.plot([beat_MHz], [w_nodrag[k]], "o", color=BLUE, ms=9, zorder=5)
ax.plot([beat_MHz], [w_drag[k]], "o", color=RED, ms=9, zorder=5)
ax.annotate("spectator", xy=(beat_MHz, 1.02), xytext=(beat_MHz, 1.08),
            ha="center", va="bottom", fontsize=12.5, color="0.35")
ax.annotate("DRAG null", xy=(beat_MHz, w_drag[k]),
            xytext=(beat_MHz - 46, 0.30), fontsize=12.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))

ax.set_xlabel(r"Detuning $\delta$  (MHz)", fontsize=16)
ax.set_ylabel(r"Normalized Spectral Weight  $|\tilde\Omega(\delta)|$", fontsize=16)
ax.set_xlim(-70, 70)
ax.set_ylim(0, 1.25)
ax.set_title("Effect of DRAG on Fourier Weight",
             fontsize=16, color=BLUE, pad=12, fontweight="bold")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(width=1.4, labelsize=13)
leg = ax.legend(loc="upper left", frameon=False, fontsize=13,
                handlelength=1.6, borderaxespad=0.5)
for txt, col in zip(leg.get_texts(), (BLUE, RED)):
    txt.set_color(col)

fig.tight_layout()
fig.savefig("drag_fourier_figure.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig("drag_fourier_figure.pdf",
            bbox_inches="tight", facecolor="white")
print("saved drag_fourier_figure.png / .pdf")