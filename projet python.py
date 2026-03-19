import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import odeint


def sir_model(state, t, transmission_rate, recovery_rate, population):
    susceptible, infected, recovered = state
    d_susceptible_dt = -transmission_rate * susceptible * infected / population
    d_infected_dt = (
        transmission_rate * susceptible * infected / population
        - recovery_rate * infected
    )
    d_recovered_dt = recovery_rate * infected
    return d_susceptible_dt, d_infected_dt, d_recovered_dt


def solve_sir(
    transmission_rate,
    recovery_rate,
    population,
    initial_infected,
    initial_recovered,
    days,
):
    initial_susceptible = population - initial_infected - initial_recovered
    initial_state = [initial_susceptible, initial_infected, initial_recovered]
    time = np.linspace(0, days, 1000)
    solution = odeint(
        sir_model,
        initial_state,
        time,
        args=(transmission_rate, recovery_rate, population),
    )
    susceptible, infected, recovered = solution.T
    return time, susceptible, infected, recovered


class SIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Application SIR")
        self.root.geometry("1100x700")

        self.inputs = {}
        self._build_layout()
        self._build_plot()
        self.update_plot()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(container, text="Parametres", padding=12)
        controls.pack(side="left", fill="y", padx=(0, 16))

        fields = [
            ("Taux de transmission (B)", "0.3"),
            ("Taux de guerison (A)", "0.1"),
            ("Population totale", "10000"),
            ("Infectes initiaux", "1"),
            ("Retablis initiaux", "0"),
            ("Duree (jours)", "100"),
        ]

        for row, (label, default_value) in enumerate(fields):
            ttk.Label(controls, text=label).grid(
                row=row, column=0, sticky="w", pady=6
            )
            entry = ttk.Entry(controls, width=18)
            entry.insert(0, default_value)
            entry.grid(row=row, column=1, sticky="ew", pady=6)
            self.inputs[label] = entry

        controls.columnconfigure(1, weight=1)

        ttk.Button(
            controls, text="Mettre a jour le graphique", command=self.update_plot
        ).grid(row=len(fields), column=0, columnspan=2, sticky="ew", pady=(12, 6))

        ttk.Button(controls, text="Reinitialiser", command=self.reset_defaults).grid(
            row=len(fields) + 1, column=0, columnspan=2, sticky="ew"
        )

        self.summary_label = ttk.Label(
            controls,
            text="",
            justify="left",
            wraplength=260,
        )
        self.summary_label.grid(
            row=len(fields) + 2, column=0, columnspan=2, sticky="w", pady=(16, 0)
        )

        self.plot_frame = ttk.Frame(container)
        self.plot_frame.pack(side="right", fill="both", expand=True)

    def _build_plot(self):
        self.figure, self.axis = plt.subplots(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def reset_defaults(self):
        defaults = {
            "Taux de transmission (B)": "0.3",
            "Taux de guerison (A)": "0.1",
            "Population totale": "10000",
            "Infectes initiaux": "1",
            "Retablis initiaux": "0",
            "Duree (jours)": "100",
        }
        for label, value in defaults.items():
            self.inputs[label].delete(0, tk.END)
            self.inputs[label].insert(0, value)
        self.update_plot()

    def update_plot(self):
        try:
            transmission_rate = float(
                self.inputs["Taux de transmission (B)"].get().strip()
            )
            recovery_rate = float(self.inputs["Taux de guerison (A)"].get().strip())
            population = int(float(self.inputs["Population totale"].get().strip()))
            initial_infected = int(
                float(self.inputs["Infectes initiaux"].get().strip())
            )
            initial_recovered = int(
                float(self.inputs["Retablis initiaux"].get().strip())
            )
            days = int(float(self.inputs["Duree (jours)"].get().strip()))
        except ValueError:
            messagebox.showerror(
                "Valeurs invalides",
                "Entre uniquement des nombres valides dans tous les champs.",
            )
            return

        if population <= 0 or days <= 0:
            messagebox.showerror(
                "Valeurs invalides",
                "La population totale et la duree doivent etre positives.",
            )
            return

        if transmission_rate < 0 or recovery_rate < 0:
            messagebox.showerror(
                "Valeurs invalides",
                "Les taux de transmission et de guerison doivent etre positifs.",
            )
            return

        if initial_infected < 0 or initial_recovered < 0:
            messagebox.showerror(
                "Valeurs invalides",
                "Les valeurs initiales ne peuvent pas etre negatives.",
            )
            return

        if initial_infected + initial_recovered >= population:
            messagebox.showerror(
                "Valeurs invalides",
                "Le total infectes + retablis doit rester inferieur a la population.",
            )
            return

        time, susceptible, infected, recovered = solve_sir(
            transmission_rate,
            recovery_rate,
            population,
            initial_infected,
            initial_recovered,
            days,
        )

        peak_index = int(np.argmax(infected))
        peak_infected = int(round(infected[peak_index]))
        peak_day = time[peak_index]

        self.axis.clear()
        self.axis.plot(time, susceptible, color="blue", label="Susceptibles")
        self.axis.plot(time, infected, color="red", label="Infectes")
        self.axis.plot(time, recovered, color="green", label="Retablis")
        self.axis.set_title("Modele SIR de propagation")
        self.axis.set_xlabel("Temps (jours)")
        self.axis.set_ylabel("Population")
        self.axis.grid(True, alpha=0.3)
        self.axis.legend()
        self.figure.tight_layout()
        self.canvas.draw()

        self.summary_label.config(
            text=(
                f"Pic des infectes : {peak_infected}\n"
                f"Jour du pic : {peak_day:.1f}\n"
                f"Retablis en fin de simulation : {int(round(recovered[-1]))}"
            )
        )



