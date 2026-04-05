import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

plt.style.use('dark_background')

DURATION = 20.0       
NUM_MATH = 100        
NUM_DISPLAY = 5       
FPS = 30              

class FourierCompleteWithGrid:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 14))
        
        gs = GridSpec(3, 2, height_ratios=[1, 1, 1.5], width_ratios=[1, 2])
        self.fig.subplots_adjust(hspace=0.4, wspace=0.15, top=0.95, bottom=0.05, left=0.05, right=0.95)

        self.ax_draw = self.fig.add_subplot(gs[0, :])
        self.ax_draw.set_title("1. INPUT SIGNAL (20s)", color='cyan', fontsize=14, fontweight='bold')
        self.ax_draw.set_xlim(0, DURATION)
        self.ax_draw.set_ylim(-4, 4)
        self.ax_draw.set_facecolor('#111111')
        
        
        ticks = np.arange(0, DURATION + 1, 2)
        self.ax_draw.set_xticks(ticks)
        self.ax_draw.set_xticklabels([f"{int(t)}s" for t in ticks])
        self.ax_draw.grid(True, color='gray', linestyle=':', alpha=0.5) 
        
        self.text_status = self.ax_draw.text(DURATION/2, 0, "DRAW HERE...", color='gray', fontsize=20, ha='center', alpha=0.5)
        self.line_draw, = self.ax_draw.plot([], [], color='cyan', linewidth=3)

        self.ax_circles = self.fig.add_subplot(gs[1, 0])
        self.ax_circles.set_title(f"2a. TOP {NUM_DISPLAY} PHASORS", color='magenta', fontsize=12)
        self.ax_circles.axis('off')
        self.ax_circles.set_aspect('equal') 
        self.ax_circles.set_xlim(-4, 4)

        self.ax_waves = self.fig.add_subplot(gs[1, 1])
        self.ax_waves.set_title("2b. TIME DOMAIN COMPONENTS", color='magenta', fontsize=12)
        self.ax_waves.set_facecolor('black')
        self.ax_waves.set_xlim(0, DURATION)
        self.ax_waves.set_xticks(ticks) 
        self.ax_waves.set_xticklabels([]) 
        self.ax_waves.grid(True, color='gray', linestyle=':', alpha=0.2, axis='x')
        self.ax_waves.axis('off') 
        
        self.ax_spec = self.fig.add_subplot(gs[2, 0])
        self.ax_spec.set_title("4. FREQUENCY SPECTRUM", color='lime', fontsize=12)
        self.ax_spec.set_facecolor('black')
        self.ax_spec.set_xlabel("Frequency (Hz)", color='gray')
        self.ax_spec.set_ylabel("Amplitude", color='gray')

        self.ax_recon = self.fig.add_subplot(gs[2, 1])
        self.ax_recon.set_title(f"3. RECONSTRUCTED SIGNAL ({NUM_MATH} Vectors)", color='yellow', fontsize=14)
        self.ax_recon.set_facecolor('black')
        self.ax_recon.set_xlim(-5, DURATION) 
        self.ax_recon.set_ylim(-4, 4)
        self.ax_recon.spines['top'].set_visible(False)
        self.ax_recon.spines['right'].set_visible(False)
        self.ax_recon.set_xlabel("Time (s)", color='gray')
        self.ax_recon.set_xticks(ticks) 
        self.ax_recon.axvline(0, color='gray', linestyle='--', alpha=0.5)

        self.raw_x, self.raw_y = [], []
        self.is_drawing = False
        self.is_analyzed = False
        self.components = []
        self.current_time = 0.0 
        self.STACK_ORIGIN_X = -3.0
        self.colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00', '#FFA500'] 

        self.p2_objs = [] 
        self.p3_vectors = [] 
        self.bar_container = None

        
        self.y_spacings = [8, 6, 4, 2, 0] 
        self.ax_circles.set_ylim(-2, 10)
        self.ax_waves.set_ylim(-2, 10)

        
        for i in range(NUM_DISPLAY):
            circ, = self.ax_circles.plot([], [], color='gray', alpha=0.3)
            ln, = self.ax_circles.plot([], [], color=self.colors[i], linewidth=2)
            dt, = self.ax_circles.plot([], [], marker='o', color=self.colors[i], markersize=5)
            txt = self.ax_circles.text(0, 0, "", color=self.colors[i], fontsize=10, fontweight='bold', ha='left')
            wv, = self.ax_waves.plot([], [], color=self.colors[i], linewidth=2, alpha=0.8)
            self.p2_objs.append({'c': circ, 'l': ln, 'd': dt, 't': txt, 'w': wv})

        # init Panel 3 (100 Vectors)
        for i in range(NUM_MATH):
            col = self.colors[i] if i < NUM_DISPLAY else 'gray'
            lw = 3 if i < NUM_DISPLAY else 1
            alpha = 0.9 if i < NUM_DISPLAY else 0.3
            vec, = self.ax_recon.plot([], [], color=col, linewidth=lw, alpha=alpha)
            self.p3_vectors.append(vec)

        self.line_recon, = self.ax_recon.plot([], [], color='white', linewidth=2, label='Reconstruction')
        self.dot_tip, = self.ax_recon.plot([], [], marker='o', color='white', markersize=6)
        self.line_projector, = self.ax_recon.plot([], [], color='yellow', linestyle='--', alpha=0.5)
        self.dot_stack_tip, = self.ax_recon.plot([], [], marker='o', color='yellow', markersize=5)
        self.line_phantom, = self.ax_recon.plot([], [], color='cyan', linewidth=1, alpha=0.3, linestyle='--', label='Original')

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.anim = FuncAnimation(self.fig, self.update, interval=1000/FPS, blit=False)
        plt.show()

    def on_press(self, event):
        if event.inaxes != self.ax_draw: return
        self.is_drawing = True
        self.is_analyzed = False
        self.raw_x = [event.xdata]
        self.raw_y = [event.ydata]
        self.line_draw.set_data(self.raw_x, self.raw_y)
        self.text_status.set_visible(False)
        self.current_time = 0.0
        
        self.line_recon.set_data([], [])
        self.line_phantom.set_data([], [])
        self.line_projector.set_data([], [])
        if self.bar_container:
            self.bar_container.remove()
            self.bar_container = None
        for i in range(NUM_MATH):
            self.p3_vectors[i].set_data([], [])

    def on_move(self, event):
        if not self.is_drawing or event.inaxes != self.ax_draw: return
        self.raw_x.append(event.xdata)
        self.raw_y.append(event.ydata)
        self.line_draw.set_data(self.raw_x, self.raw_y)

    def on_release(self, event):
        if not self.is_drawing: return
        self.is_drawing = False
        self.analyze()

    def analyze(self):
        if len(self.raw_x) < 5: return
        
        sorted_pairs = sorted(zip(self.raw_x, self.raw_y))
        sx = [p[0] for p in sorted_pairs]
        sy = [p[1] for p in sorted_pairs]
        ux, idx = np.unique(sx, return_index=True)
        uy = np.array(sy)[idx]
        
        t_uniform = np.linspace(0, DURATION, 2000)
        f_interp = interp1d(ux, uy, kind='linear', fill_value=0, bounds_error=False)
        y_uniform = f_interp(t_uniform)
        self.phantom_t, self.phantom_y = t_uniform, y_uniform
        
        fft_vals = np.fft.rfft(y_uniform)
        fft_freqs = np.fft.rfftfreq(len(t_uniform), d=t_uniform[1]-t_uniform[0])
        mags = np.abs(fft_vals) / len(t_uniform) * 2
        phases = np.angle(fft_vals)
        mags[0] = 0 
        
        # Spectrum
        self.ax_spec.cla()
        self.ax_spec.set_title("4. FREQUENCY SPECTRUM", color='lime', fontsize=12)
        self.ax_spec.set_facecolor('black')
        self.ax_spec.set_xlabel("Frequency (Hz)", color='gray')
        self.ax_spec.set_ylabel("Amplitude", color='gray')
        
        mask = (fft_freqs > 0) & (fft_freqs < 5.0)
        plot_freqs = fft_freqs[mask]
        plot_mags = mags[mask]
        self.bar_container = self.ax_spec.bar(plot_freqs, plot_mags, width=0.05, color='lime', alpha=0.7)
        self.ax_spec.set_xlim(0, 3.0) 
        self.ax_spec.set_ylim(0, max(plot_mags)*1.1)

        top_indices = np.argsort(mags)[-NUM_MATH:][::-1]
        
        self.components = []
        for i, idx in enumerate(top_indices):
            self.components.append({'f': fft_freqs[idx], 'a': mags[idx], 'p': phases[idx]})
            if i < NUM_DISPLAY:
                self.ax_spec.plot(fft_freqs[idx], mags[idx], 'o', color=self.colors[i], markersize=8)
                y_off = self.y_spacings[i]
                theta = np.linspace(0, 2*np.pi, 50)
                r = mags[idx]
                self.p2_objs[i]['c'].set_data(r*np.cos(theta), y_off + r*np.sin(theta))
                self.p2_objs[i]['t'].set_position((2.5, y_off))
                self.p2_objs[i]['t'].set_text(f"{fft_freqs[idx]:.2f} Hz")
            
        self.is_analyzed = True
        self.current_time = 0.0

    def update(self, frame):
        if not self.is_analyzed: return
        
        self.current_time += 1/FPS
        if self.current_time > DURATION: self.current_time = 0.0
        t = self.current_time
        
        history_t = np.linspace(0, t, int(t * 30) + 1)
        combined_signal = np.zeros_like(history_t)
        
        curr_x = self.STACK_ORIGIN_X
        curr_y = 0
        
        for i, c in enumerate(self.components):
            angle = 2 * np.pi * c['f'] * t + c['p']
            
            if i < NUM_DISPLAY:
                y_off = self.y_spacings[i]
                dx, dy = c['a'] * np.cos(angle), c['a'] * np.sin(angle)
                self.p2_objs[i]['l'].set_data([0, dx], [y_off, y_off + dy])
                self.p2_objs[i]['d'].set_data([dx], [y_off + dy])
                wave_y = c['a'] * np.sin(2 * np.pi * c['f'] * history_t + c['p'])
                self.p2_objs[i]['w'].set_data(history_t, y_off + wave_y)
            
            combined_signal += c['a'] * np.sin(2 * np.pi * c['f'] * history_t + c['p'])
            dx, dy = c['a'] * np.cos(angle), c['a'] * np.sin(angle)
            self.p3_vectors[i].set_data([curr_x, curr_x+dx], [curr_y, curr_y+dy])
            curr_x += dx
            curr_y += dy
            
        self.line_projector.set_data([curr_x, t], [curr_y, curr_y])
        self.dot_stack_tip.set_data([curr_x], [curr_y])
        self.line_recon.set_data(history_t, combined_signal)
        self.dot_tip.set_data([t], [curr_y])
        
        return [self.line_recon, self.dot_tip]

if __name__ == "__main__":
    app = FourierCompleteWithGrid()