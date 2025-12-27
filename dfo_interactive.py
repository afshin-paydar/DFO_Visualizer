"""
DFO Visualizer
==========================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
warnings.filterwarnings('ignore')


class DFO:
    """DFO Algorithm Implementation"""

    def __init__(self, objective_func, n_flies=30, dimensions=2,
                 delta=0.01, bounds=(-5, 5), max_iterations=100):
        self.f = objective_func
        self.N = n_flies
        self.D = dimensions
        self.delta = delta
        self.bounds = bounds
        self.max_iter = max_iterations

        # Initialize
        self.reset()

    def reset(self):
        """Reset the optimizer"""
        self.X = np.random.uniform(self.bounds[0], self.bounds[1], (self.N, self.D))
        self.fitness = np.ones(self.N) * np.inf
        self.best_position = None
        self.best_fitness = np.inf
        self.best_fitness_history = []
        self.iteration = 0

    def evaluate_fitness(self):
        """Evaluate fitness for all flies"""
        for i in range(self.N):
            try:
                self.fitness[i] = self.f(self.X[i])
            except:
                self.fitness[i] = np.inf

        # Update global best
        current_best_idx = np.argmin(self.fitness)
        if self.fitness[current_best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[current_best_idx]
            self.best_position = self.X[current_best_idx].copy()

    def find_best_neighbor(self, i):
        """Find best neighbor in ring topology"""
        left_idx = (i - 1) % self.N
        right_idx = (i + 1) % self.N
        return left_idx if self.fitness[left_idx] < self.fitness[right_idx] else right_idx

    def update_positions(self):
        """Update fly positions"""
        s = np.argmin(self.fitness)
        X_new = self.X.copy()

        for i in range(self.N):
            if i == s:  # Skip best fly (elitist strategy)
                continue

            best_neighbor = self.find_best_neighbor(i)

            for d in range(self.D):
                if np.random.rand() < self.delta:
                    # Disturbance
                    X_new[i, d] = np.random.uniform(self.bounds[0], self.bounds[1])
                else:
                    # DFO update equation
                    u = np.random.rand()
                    X_new[i, d] = self.X[best_neighbor, d] + u * (self.X[s, d] - self.X[i, d])

                # Enforce bounds
                if X_new[i, d] < self.bounds[0] or X_new[i, d] > self.bounds[1]:
                    X_new[i, d] = np.random.uniform(self.bounds[0], self.bounds[1])

        self.X = X_new

    def step(self):
        """Perform one iteration"""
        self.evaluate_fitness()
        self.best_fitness_history.append(self.best_fitness)
        self.update_positions()
        self.iteration += 1


class DFOVisualizer:
    """DFO visualization"""

    def __init__(self, root):
        self.root = root
        self.root.title("DFO Visualizer")
        self.root.geometry("1400x800")

        # Default function
        self.current_function = lambda x: x[0]**2 + x[1]**2
        self.function_string = "x[0]**2 + x[1]**2"

        # DFO instance
        self.dfo = None
        self.is_running = False
        self.animation_id = None

        # Setup GUI
        self.setup_gui()

        # Initialize
        self.reset_optimization()

    def setup_gui(self):
        """Setup the GUI layout"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        row = 0

        # Predefined functions
        ttk.Label(control_frame, text="Function:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        self.function_var = tk.StringVar(value="Sphere (Easy)")
        functions = [
            "Sphere (Easy)",
            "Rosenbrock (Medium)",
            "Rastrigin (Hard - Multimodal)",
            "Ackley (Hard - Multimodal)",
            "Beale (Medium)",
            "Himmelblau (Medium)",
            "Custom"
        ]
        function_dropdown = ttk.Combobox(control_frame, textvariable=self.function_var,
                                         values=functions, state="readonly", width=25)
        function_dropdown.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        function_dropdown.bind("<<ComboboxSelected>>", self.on_function_select)
        row += 1

        # Add difficulty hint
        self.difficulty_label = ttk.Label(control_frame, text="Easy: Low N works well",
                                         font=('Arial', 8, 'italic'), foreground='green')
        self.difficulty_label.grid(row=row, column=0, sticky=tk.W)
        row += 1

        # Custom function input
        ttk.Label(control_frame, text="Custom f(x,y):").grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        self.function_entry = tk.Text(control_frame, height=3, width=30)
        self.function_entry.insert("1.0", self.function_string)
        self.function_entry.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1

        ttk.Label(control_frame, text="Use: x[0], x[1], np functions",
                 font=('Arial', 8, 'italic')).grid(row=row, column=0, sticky=tk.W)
        row += 1

        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Parameters
        ttk.Label(control_frame, text="Parameters:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        # N_FLIES
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(param_frame, text="N_FLIES:", width=12).pack(side=tk.LEFT)
        self.n_flies_var = tk.IntVar(value=30)
        self.n_flies_label = ttk.Label(param_frame, text="30", width=5)
        self.n_flies_label.pack(side=tk.RIGHT)
        row += 1

        n_flies_slider = ttk.Scale(control_frame, from_=10, to=100, orient=tk.HORIZONTAL,
                                   variable=self.n_flies_var,
                                   command=lambda v: self.n_flies_label.config(text=str(int(float(v)))))
        n_flies_slider.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        row += 1

        # DELTA
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(param_frame, text="DELTA:", width=12).pack(side=tk.LEFT)
        self.delta_var = tk.DoubleVar(value=0.01)
        self.delta_label = ttk.Label(param_frame, text="0.010", width=5)
        self.delta_label.pack(side=tk.RIGHT)
        row += 1

        delta_slider = ttk.Scale(control_frame, from_=0.001, to=0.2, orient=tk.HORIZONTAL,
                                variable=self.delta_var,
                                command=lambda v: self.delta_label.config(text=f"{float(v):.3f}"))
        delta_slider.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        row += 1

        # MAX_ITERATIONS
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(param_frame, text="Iterations:", width=12).pack(side=tk.LEFT)
        self.max_iter_var = tk.IntVar(value=100)
        self.max_iter_label = ttk.Label(param_frame, text="100", width=5)
        self.max_iter_label.pack(side=tk.RIGHT)
        row += 1

        max_iter_slider = ttk.Scale(control_frame, from_=50, to=500, orient=tk.HORIZONTAL,
                                    variable=self.max_iter_var,
                                    command=lambda v: self.max_iter_label.config(text=str(int(float(v)))))
        max_iter_slider.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        row += 1

        # BOUNDS
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(param_frame, text="Bounds:").pack(side=tk.LEFT)
        self.bounds_min_var = tk.DoubleVar(value=-5.0)
        ttk.Entry(param_frame, textvariable=self.bounds_min_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(param_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.bounds_max_var = tk.DoubleVar(value=5.0)
        ttk.Entry(param_frame, textvariable=self.bounds_max_var, width=6).pack(side=tk.LEFT, padx=2)
        row += 1

        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=row, column=0, pady=10)

        self.start_button = ttk.Button(button_frame, text="▶ Start",
                                       command=self.start_optimization, width=10)
        self.start_button.pack(side=tk.LEFT, padx=3)

        self.stop_button = ttk.Button(button_frame, text="⏸ Pause",
                                      command=self.stop_optimization, width=10, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=3)

        self.reset_button = ttk.Button(button_frame, text="↻ Reset",
                                       command=self.reset_optimization, width=10)
        self.reset_button.pack(side=tk.LEFT, padx=3)
        row += 1

        # Animation speed
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        ttk.Label(param_frame, text="Speed (ms):", width=12).pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=50)
        self.speed_label = ttk.Label(param_frame, text="50", width=5)
        self.speed_label.pack(side=tk.RIGHT)
        row += 1

        speed_slider = ttk.Scale(control_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                                variable=self.speed_var,
                                command=lambda v: self.speed_label.config(text=str(int(float(v)))))
        speed_slider.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        row += 1

        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Status
        ttk.Label(control_frame, text="Status:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        self.status_text = tk.Text(control_frame, height=8, width=30, font=('Courier', 9))
        self.status_text.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)

        # Right panel - Visualization
        viz_frame = ttk.Frame(main_frame)
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Create figure
        self.fig = Figure(figsize=(11, 8))
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2, projection='3d')
        self.ax3 = self.fig.add_subplot(2, 2, (3, 4))

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def on_function_select(self, event):
        """Handle function selection"""
        funcs = {
            "Sphere (Easy)": "x[0]**2 + x[1]**2",
            "Rosenbrock (Medium)": "(1-x[0])**2 + 100*(x[1]-x[0]**2)**2",
            "Rastrigin (Hard - Multimodal)": "10*2 + x[0]**2 - 10*np.cos(2*np.pi*x[0]) + x[1]**2 - 10*np.cos(2*np.pi*x[1])",
            "Ackley (Hard - Multimodal)": "-20*np.exp(-0.2*np.sqrt((x[0]**2+x[1]**2)/2)) - np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2) + 20 + np.e",
            "Beale (Medium)": "(1.5-x[0]+x[0]*x[1])**2 + (2.25-x[0]+x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2",
            "Himmelblau (Medium)": "(x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2",
            "Custom": self.function_string
        }

        difficulty_hints = {
            "Sphere (Easy)": ("Easy: Low N works well", 'green'),
            "Rosenbrock (Medium)": ("Medium: Use N=20-30", 'orange'),
            "Rastrigin (Hard - Multimodal)": ("Hard: Use N=30-50, DELTA=0.02-0.05", 'red'),
            "Ackley (Hard - Multimodal)": ("Hard: Use N=30-50, DELTA=0.02-0.05", 'red'),
            "Beale (Medium)": ("Medium: Use N=20-30", 'orange'),
            "Himmelblau (Medium)": ("Medium: Use N=20-30", 'orange'),
            "Custom": ("Custom: Adjust params as needed", 'blue')
        }

        selected = self.function_var.get()
        self.function_entry.delete("1.0", tk.END)
        self.function_entry.insert("1.0", funcs.get(selected, ""))

        # Update difficulty hint
        if selected in difficulty_hints:
            hint_text, hint_color = difficulty_hints[selected]
            self.difficulty_label.config(text=hint_text, foreground=hint_color)

    def parse_function(self):
        """Parse the user's function"""
        func_str = self.function_entry.get("1.0", tk.END).strip()
        if not func_str:
            messagebox.showerror("Error", "Please enter a function!")
            return None

        try:
            def user_function(x):
                return eval(func_str, {"np": np, "x": x, "sin": np.sin, "cos": np.cos,
                                       "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
                                       "log": np.log, "tan": np.tan, "pi": np.pi})

            # Test
            test_point = np.array([0.0, 0.0])
            result = user_function(test_point)
            if not np.isfinite(result):
                raise ValueError("Function returned inf or nan")

            self.function_string = func_str
            return user_function

        except Exception as e:
            messagebox.showerror("Function Error", f"Invalid function:\n{str(e)}")
            return None

    def reset_optimization(self):
        """Reset optimization"""
        self.stop_optimization()

        func = self.parse_function()
        if func is None:
            return

        self.current_function = func

        # Create DFO
        self.dfo = DFO(
            objective_func=self.current_function,
            n_flies=self.n_flies_var.get(),
            dimensions=2,
            delta=self.delta_var.get(),
            bounds=(self.bounds_min_var.get(), self.bounds_max_var.get()),
            max_iterations=self.max_iter_var.get()
        )

        # Setup visualization
        self.setup_visualization()
        self.update_status()

    def setup_visualization(self):
        """Setup visualization"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Create surface
        bounds = (self.bounds_min_var.get(), self.bounds_max_var.get())
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[0], bounds[1], 100)
        self.X_mesh, self.Y_mesh = np.meshgrid(x, y)

        self.Z_mesh = np.zeros_like(self.X_mesh)
        for i in range(self.X_mesh.shape[0]):
            for j in range(self.X_mesh.shape[1]):
                try:
                    self.Z_mesh[i, j] = self.current_function(np.array([self.X_mesh[i, j], self.Y_mesh[i, j]]))
                except:
                    self.Z_mesh[i, j] = np.nan

        # Contour plot
        self.ax1.contourf(self.X_mesh, self.Y_mesh, self.Z_mesh, levels=30, cmap='viridis', alpha=0.6)
        self.ax1.contour(self.X_mesh, self.Y_mesh, self.Z_mesh, levels=15,
                        colors='black', alpha=0.3, linewidths=0.5)

        positions = self.dfo.X
        self.scatter1 = self.ax1.scatter(positions[:, 0], positions[:, 1],
                                        c='red', s=60, alpha=0.7, edgecolors='black', linewidths=1.5)
        self.best_scatter1 = self.ax1.scatter([], [], c='gold', s=250, marker='*',
                                             edgecolors='black', linewidths=2, zorder=10)
        self.ax1.set_xlabel('x', fontsize=10)
        self.ax1.set_ylabel('y', fontsize=10)
        self.ax1.set_title('Contour View', fontsize=11, weight='bold')
        self.ax1.grid(True, alpha=0.3)

        # 3D plot
        self.ax2.plot_surface(self.X_mesh, self.Y_mesh, self.Z_mesh,
                             cmap='viridis', alpha=0.6, edgecolor='none')
        z_pos = np.array([self.current_function(p) for p in positions])
        self.scatter2 = self.ax2.scatter(positions[:, 0], positions[:, 1], z_pos,
                                        c='red', s=60, alpha=0.8, edgecolors='black', linewidths=1.5)
        self.best_scatter2 = self.ax2.scatter([], [], [], c='gold', s=150, marker='*',
                                             edgecolors='black', linewidths=2, zorder=10)
        self.ax2.set_xlabel('x', fontsize=9)
        self.ax2.set_ylabel('y', fontsize=9)
        self.ax2.set_zlabel('f(x,y)', fontsize=9)
        self.ax2.set_title('3D Surface', fontsize=11, weight='bold')

        # Convergence plot
        self.line3, = self.ax3.plot([], [], 'b-', linewidth=2)
        self.ax3.set_xlabel('Iteration', fontsize=8)
        self.ax3.set_ylabel('Best Fitness', fontsize=8)
        self.ax3.set_title('Convergence Curve', fontsize=10, weight='bold')
        self.ax3.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_visualization(self):
        """Update visualization"""
        if self.dfo is None or self.dfo.best_position is None:
            return

        # Update contour scatter
        positions = self.dfo.X
        self.scatter1.set_offsets(positions)
        self.best_scatter1.set_offsets(self.dfo.best_position.reshape(1, -1))

        # Update 3D scatter
        z_pos = np.array([self.current_function(p) for p in positions])
        self.scatter2._offsets3d = (positions[:, 0], positions[:, 1], z_pos)
        self.best_scatter2._offsets3d = ([self.dfo.best_position[0]],
                                        [self.dfo.best_position[1]],
                                        [self.dfo.best_fitness])

        # Update convergence
        if len(self.dfo.best_fitness_history) > 0:
            iters = list(range(len(self.dfo.best_fitness_history)))
            self.line3.set_data(iters, self.dfo.best_fitness_history)
            self.ax3.relim()
            self.ax3.autoscale_view()

        self.canvas.draw()
        self.update_status()

    def update_status(self):
        """Update status text"""
        if self.dfo is None:
            status = "No optimization"
        else:
            status = f"Iteration: {self.dfo.iteration}/{self.dfo.max_iter}\n"
            status += f"Best Fitness: {self.dfo.best_fitness:.8f}\n"
            status += f"Population: {self.dfo.N} flies\n"
            status += f"Delta: {self.dfo.delta:.4f}\n"
            status += f"Status: {'Running' if self.is_running else 'Stopped'}\n"

            if self.dfo.best_position is not None:
                status += f"\nBest Position:\n"
                status += f"  x = {self.dfo.best_position[0]:.6f}\n"
                status += f"  y = {self.dfo.best_position[1]:.6f}"

        self.status_text.delete("1.0", tk.END)
        self.status_text.insert("1.0", status)

    def start_optimization(self):
        """Start optimization"""
        if self.dfo is None:
            self.reset_optimization()

        if self.is_running:
            return

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.DISABLED)

        self.run_step()

    def stop_optimization(self):
        """Stop optimization"""
        self.is_running = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.NORMAL)

    def run_step(self):
        """Run one optimization step"""
        if not self.is_running or self.dfo is None:
            return

        if self.dfo.iteration < self.dfo.max_iter:
            # Perform DFO step
            self.dfo.step()

            # Update visualization
            self.update_visualization()

            # Schedule next step
            self.animation_id = self.root.after(self.speed_var.get(), self.run_step)
        else:
            # Optimization complete
            self.stop_optimization()
            messagebox.showinfo("Complete",
                f"Optimization finished!\n\nBest fitness: {self.dfo.best_fitness:.8f}\n" +
                f"Best position: ({self.dfo.best_position[0]:.4f}, {self.dfo.best_position[1]:.4f})")


if __name__ == "__main__":
    root = tk.Tk()
    app = DFOVisualizer(root)
    root.mainloop()
