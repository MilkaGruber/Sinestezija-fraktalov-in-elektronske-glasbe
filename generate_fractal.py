# generate_fractal.py - BOLJŠA KAKOVOST IN VEČ OKVIRJEV
import pygame
import numpy as np
import math
import random
import colorsys
import librosa
import json
import os
import sys
import pickle
from datetime import datetime
from collections import deque

##################################################################################
####################### Analiza zvoka ############################################
##################################################################################

# Inicializacija pygame - uporabimo za predvajanje zvoka
try:
    pygame.init()
    # Poskusi inicializirati audio-mixer
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    except:
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
        except:
            try:
                pygame.mixer.init()
            except:
                print("Audio-mixer ni inicializiran - predvajanje brez zvoka")
    print("Pygame uspešno inicializiran")
except Exception as e:
    print(f"Napaka pri inicializaciji pygame: {e}")

class AudioAnalyzer:
    """Razred, v katerem analiziramo glasbeno datoteko."""
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.analyzed_data = None
        
    def analyze_audio(self):
        """Funkcija, ki iz glasbene datoteke najde
            - 'duration': float (trajanje skladbe v sekundah)
            - 'sample_rate': int (vzorčna frekvenca v Hz)
            - 'tempo': float  (tempo skladbe v BPM)
            - 'beat_times': list of float (časi beatov v sekundah)
            - 'onset_times': list of float (časi začetkov posameznih zvokov v sekundah)
            - 'rms': list of float (normalizirana RMS po segmentih)
            - 'rms_times': list of float (časi posameznih RMS segmentov)
            - 'spectral_centroid': list of float (normaliziran spektralni centroid po segmentih)
            - 'centroid_times': list of float (časi segmentov spektralnega centroida)
            - 'mel_spectrogram': list of list of float (normaliziran Mel spektrogram) 
            - 'total_samples': int (skupno število vzorcev v audio datoteki)
            - 'audio_file': str (ime/audio datoteka, ki je bila analizirana)
        """
        print(f"Analiziram glasbo: {self.audio_file}")
        
        try:
            # Naloži glasbo
            y, sr = librosa.load(self.audio_file, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            
            print(f"Naloženo: {duration:.2f}s, {sr}Hz, {len(y)} vzorcev")
            
            # Izračunaj spektrogram (osnovna frekvenčna analiza)
            print("Izračunavam spektrogram...")
            S = np.abs(librosa.stft(y))
            
            # Beat tracking
            print("Analiza beatov...")
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Če je tempo array, vzemi prvi element
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else 120.0
            
            # Onset detection (zaznavanje nastopa novih zvokov) - marker za spremembo
            print("Iščem onsete...")
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Izračunaj volumen (Root Mean Square)
            print("Izračunavam volumen...")
            hop_length = 512 # dolžina koraka na katerem računamo "povprečje"
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            rms_times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
            
            # Normaliziraj RMS
            if len(rms) > 0:
                rms_max = np.max(rms)
                if rms_max > 0:
                    rms = rms / rms_max
            
            # Izračunaj spektralni centroid
            print("  Izračunavam spektralne značilnosti...")
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_times = librosa.frames_to_time(range(len(spectral_centroid)), sr=sr)
            
            # Normaliziraj centroid
            if len(spectral_centroid) > 0:
                centroid_max = np.max(spectral_centroid)
                if centroid_max > 0:
                    spectral_centroid = spectral_centroid / centroid_max
            
            # Izračunaj mel frekvence
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # VEČ mel frekvenc za boljšo kakovost
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normaliziraj mel spektrogram
            if np.max(mel_spec_db) > np.min(mel_spec_db):
                mel_spec_db_normalized = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
            else:
                mel_spec_db_normalized = mel_spec_db
            
            # Shrani vse podatke
            self.analyzed_data = {
                'duration': float(duration),
                'sample_rate': sr,
                'tempo': float(tempo),
                'beat_times': beat_times.tolist(),
                'onset_times': onset_times.tolist(),
                'rms': rms.tolist(),
                'rms_times': rms_times.tolist(),
                'spectral_centroid': spectral_centroid.tolist(),
                'centroid_times': centroid_times.tolist(),
                'mel_spectrogram': mel_spec_db_normalized.tolist(),
                'total_samples': len(y),
                'audio_file': self.audio_file
            }
            
            print(f"Analiza končana. Trajanje: {duration:.2f}s, Tempo: {tempo:.1f} BPM")
            print(f"# Beat-ov: {len(beat_times)}, # Onset-ov: {len(onset_times)}")
            return self.analyzed_data
            
        except Exception as e:
            print(f"Napaka pri analizi glasbe: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_audio_features_at_time(self, time_sec):
        """Vrne slovar značilnosti glasbe v danem trenutku:
            - 'time': float (čas v sekundah)
            - 'volume': float (glasnost (RMS) v trenutku)
            - 'brightness': float (spektralni centroid (barva zvoka) v trenutku)
            - 'is_beat': bool (ali je v tem trenutku beat)
            - 'is_onset': bool (ali se začne nov zvok (onset))
            - 'low_energy': float (energija nizkih frekvenc)
            - 'mid_energy': float (energija srednjih frekvenc)
            - 'high_energy': float (energija visokih frekvenc)
            - 'combined_energy': float (povprečna energija vseh frekvenc)
            - 'progress': float (delež skladbe, ki je že pretekel (0.0–1.0))
        """
        if self.analyzed_data is None:
            print("Glasba ni bila zaznana. Poženem funkcijo analyze_audio...")
            if not self.analyze_audio():
                return self.get_default_features(time_sec)
            
        data = self.analyzed_data
        
        # Preverimo, če je dan čas v pravih časovnih mejah
        time_sec = max(0, min(time_sec, data['duration'] - 0.1))
        
        # Najde najbljižni indeks za trenutek, ki nas zanima
        def find_nearest_idx(array, value):
            return np.argmin(np.abs(np.array(array) - value))
        
        # RMS (volumen)
        if len(data['rms']) > 0:
            rms_idx = find_nearest_idx(data['rms_times'], time_sec)
            volume = data['rms'][rms_idx]
        else:
            volume = 0.5
        
        # Spektralni centroid (barva)
        if len(data['spectral_centroid']) > 0:
            centroid_idx = find_nearest_idx(data['centroid_times'], time_sec)
            brightness = data['spectral_centroid'][centroid_idx]
        else:
            brightness = 0.5
        
        # Preveri, ali je beat
        beat_threshold = 0.15  # 150ms tolerance
        is_beat = False
        if data['beat_times']:
            is_beat = np.any(np.abs(np.array(data['beat_times']) - time_sec) < beat_threshold)
        
        # Preveri, ali je onset
        is_onset = False
        if data['onset_times']:
            is_onset = np.any(np.abs(np.array(data['onset_times']) - time_sec) < beat_threshold)
        
        # Analiza mel spektrograma za frekvenčne pasove
        mel_data = np.array(data['mel_spectrogram'])
        if mel_data.size > 0:
            mel_idx = int((time_sec / data['duration']) * mel_data.shape[1])
            mel_idx = max(0, min(mel_idx, mel_data.shape[1] - 1))
            
            # Energija v različnih frekvenčnih pasovih
            n_mels = mel_data.shape[0]
            low_freq_energy = np.mean(mel_data[:n_mels//4, mel_idx]) if n_mels//4 > 0 else 0.5
            mid_freq_energy = np.mean(mel_data[n_mels//4:3*n_mels//4, mel_idx]) if n_mels//2 > 0 else 0.5
            high_freq_energy = np.mean(mel_data[3*n_mels//4:, mel_idx]) if n_mels//4 > 0 else 0.5
            combined_energy = np.mean(mel_data[:, mel_idx])
        else:
            low_freq_energy = mid_freq_energy = high_freq_energy = combined_energy = 0.5
        
        return {
            'time': time_sec,
            'volume': float(volume),
            'brightness': float(brightness),
            'is_beat': bool(is_beat),
            'is_onset': bool(is_onset),
            'low_energy': float(low_freq_energy),
            'mid_energy': float(mid_freq_energy),
            'high_energy': float(high_freq_energy),
            'combined_energy': float(combined_energy),
            'progress': time_sec / data['duration']
        }
    
    def get_default_features(self, time_sec):
        """Vrne privzete značilnosti, če analiza ne uspe"""
        return {
            'time': time_sec,
            'volume': 0.5 + 0.3 * math.sin(time_sec * 0.5),
            'brightness': 0.5 + 0.3 * math.sin(time_sec * 0.3),
            'is_beat': int(time_sec * 2) % 4 == 0,
            'is_onset': int(time_sec * 3) % 5 == 0,
            'low_energy': 0.5 + 0.2 * math.sin(time_sec * 0.4),
            'mid_energy': 0.5 + 0.2 * math.sin(time_sec * 0.6),
            'high_energy': 0.5 + 0.2 * math.sin(time_sec * 0.8),
            'combined_energy': 0.5,
            'progress': 0.0
        }

##################################################################################
####################### Risanje fraktalov ########################################
##################################################################################

class FractalGenerator:
    """Razred za generiranje različnih vrst fraktalov - BOLJŠA KAKOVOST"""
    
    def __init__(self, width, height, quality="high"):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.scale = 300  # Večji scale za boljše fraktale
        self.quality = quality  # "low", "medium", "high"
        
        # Nastavitve števila iteracij glede na željeno kakovost
        if quality == "low":
            self.base_iterations = 30
            self.particle_count = 60
            self.pattern_elements = 20
        elif quality == "medium":
            self.base_iterations = 60
            self.particle_count = 100
            self.pattern_elements = 35
        else:  # high
            self.base_iterations = 100
            self.particle_count = 150
            self.pattern_elements = 50
    
    def mandelbrot(self, features, time, offset=0):
        """Ustvari Mandelbrotov fraktal z BOLJŠO KAKOVOSTJO"""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Parametri glede na glasbo - BOLJ IZRAZITI
        zoom = 0.8 + features['volume'] * 2.5  # Boljši zoom
        color_shift = features['brightness'] * 150
        max_iter = int(self.base_iterations + features['combined_energy'] * 100)  # VEČ iteracij
        
        # Rotacija glede na čas
        angle = time * 0.2 + offset  # Počasnejša rotacija
        
        # Kakovost glede na nastavitev
        if self.quality == "low":
            step = max(1, int(4 - features['volume']))
        elif self.quality == "medium":
            step = max(1, int(3 - features['volume'] * 0.7))
        else:  # high
            step = max(1, int(2 - features['volume'] * 0.5))
        
        # Barvna paleta za boljše barve
        palette = []
        for i in range(max_iter):
            hue = (i * 2 + color_shift + time * 10) % 360 / 360
            saturation = 0.7 + features['volume'] * 0.3
            value = 0.3 + 0.7 * (i / max_iter)
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            palette.append((int(r * 255), int(g * 255), int(b * 255)))
        
        for x in range(0, self.width, step):
            for y in range(0, self.height, step):
                # Transformacija koordinat
                nx = (x - self.center_x) / (self.scale * zoom)
                ny = (y - self.center_y) / (self.scale * zoom)
                
                # Rotacija
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                rx = nx * cos_a - ny * sin_a
                ry = nx * sin_a + ny * cos_a
                
                # Mandelbrotov algoritem
                c = complex(rx, ry)
                z = 0
                iteration = 0
                
                for i in range(max_iter):
                    if abs(z) > 2:
                        break
                    z = z * z + c
                    iteration += 1
                
                # Barvanje z boljšimi barvami
                if iteration < max_iter:
                    # Barva iz palete
                    color_idx = min(iteration, len(palette) - 1)
                    color = palette[color_idx]
                    
                    # Intenzivnost glede na glasbo
                    alpha = 200 + int(55 * features['volume'])
                    if features['is_beat']:
                        alpha = 255
                        # Dodaj sijaj za beat
                        glow_radius = step + 2
                        pygame.draw.circle(surface, 
                                         (*color, alpha // 2),
                                         (x, y), glow_radius)
                    
                    color_with_alpha = (*color, alpha)
                    
                    # Nariši
                    radius = max(1, int(step * 0.8))
                    pygame.draw.circle(surface, color_with_alpha, (x, y), radius)
                    
                    # Dodaj piko v sredino za boljši videz
                    if radius > 2:
                        center_color = (min(255, color[0] + 50), 
                                      min(255, color[1] + 50), 
                                      min(255, color[2] + 50), 
                                      alpha)
                        pygame.draw.circle(surface, center_color, (x, y), max(1, radius // 3))
        
        return surface
    
    def julia(self, features, time, offset=0):
        """Ustvari Juliajev fraktal z BOLJŠO KAKOVOSTJO"""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Juliajev parameter glede na glasbo - bolj kompleksen
        c_real = math.cos(time * 0.15 + offset) * 0.8 + features['low_energy'] * 0.2
        c_imag = math.sin(time * 0.12 + offset) * 0.8 + features['high_energy'] * 0.2
        
        # Učinki glede na glasbo
        zoom = 0.7 + features['volume'] * 2.0
        color_speed = 5 + features['brightness'] * 10  # Počasnejše spreminjanje barv
        
        # Kakovost
        if self.quality == "low":
            step = max(1, int(4 - features['volume'] * 0.5))
            max_iter = 50 + int(features['combined_energy'] * 50)
        elif self.quality == "medium":
            step = max(1, int(3 - features['volume'] * 0.3))
            max_iter = 80 + int(features['combined_energy'] * 70)
        else:  # high
            step = max(1, int(2 - features['volume'] * 0.2))
            max_iter = 120 + int(features['combined_energy'] * 100)
        
        # Barvna paleta
        palette = []
        for i in range(max_iter):
            hue = (time * color_speed + i * 1.5) % 360 / 360
            saturation = 0.8
            value = 0.4 + 0.6 * (i / max_iter)
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            palette.append((int(r * 255), int(g * 255), int(b * 255)))
        
        for x in range(0, self.width, step):
            for y in range(0, self.height, step):
                # Transformacija koordinat
                zx = 1.5 * (x - self.width / 2) / (0.5 * zoom * self.width)
                zy = (y - self.height / 2) / (0.5 * zoom * self.height)
                
                iteration = 0
                
                while zx * zx + zy * zy < 4 and iteration < max_iter:
                    xtemp = zx * zx - zy * zy + c_real
                    zy = 2 * zx * zy + c_imag
                    zx = xtemp
                    iteration += 1
                
                if iteration < max_iter:
                    # Barva iz palete
                    color_idx = min(iteration, len(palette) - 1)
                    color = palette[color_idx]
                    
                    # Intenzivnost
                    alpha = 180 + int(75 * features['volume'])
                    if features['is_onset']:
                        alpha = 255
                    
                    color_with_alpha = (*color, alpha)
                    
                    # Nariši
                    radius = max(1, int(step * 0.7))
                    pygame.draw.circle(surface, color_with_alpha, (x, y), radius)
                    
                    # Dodaj teksturo
                    if radius > 1 and iteration % 3 == 0:
                        texture_color = (min(255, color[0] + 30), 
                                       min(255, color[1] + 30), 
                                       min(255, color[2] + 30), 
                                       alpha // 2)
                        pygame.draw.circle(surface, texture_color, (x, y), max(1, radius // 2))
        
        return surface
    
    def particle_system(self, features, time, particle_count=None):
        """Ustvari sistem delcev z BOLJŠO KAKOVOSTJO"""
        if particle_count is None:
            particle_count = self.particle_count
            
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Število delcev glede na glasbo
        active_particles = int(particle_count * (0.3 + features['volume'] * 0.7))
        
        for i in range(active_particles):
            # Položaj glede na glasbo - bolj kompleksen
            angle = (i / max(1, active_particles)) * math.pi * 2 + time * 0.3
            radius = 80 + features['low_energy'] * 150 + math.sin(time * 1.2 + i * 0.05) * 40
            
            # Dodaj eliptično gibanje
            ellipse_x = radius * (1 + 0.3 * math.sin(time * 0.5 + i * 0.02))
            ellipse_y = radius * (1 + 0.3 * math.cos(time * 0.5 + i * 0.02))
            
            x = self.center_x + math.cos(angle) * ellipse_x
            y = self.center_y + math.sin(angle) * ellipse_y
            
            # Velikost delca - bolj dinamična
            size = 1.0 + features['volume'] * 4 + math.sin(time * 3 + i * 0.1) * 0.5
            if features['is_beat']:
                size *= 2.0
            
            # Barva glede na glasbo - bolj kompleksna
            hue = (features['brightness'] * 0.3 + i * 0.002 + time * 0.02) % 1.0
            saturation = 0.7 + features['mid_energy'] * 0.3
            value = 0.7 + features['volume'] * 0.3
            
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Nariši delec z boljšimi učinki
            alpha = 180 + int(75 * features['volume'])
            if features['is_onset']:
                alpha = 255
                # Boljši sev
                glow_size = int(size * 3)
                for j in range(glow_size, 0, -1):
                    glow_alpha = int(alpha * (1 - j/glow_size) * 0.2)
                    pygame.draw.circle(surface, 
                                     (int(r * 255), int(g * 255), int(b * 255), glow_alpha),
                                     (int(x), int(y)), j)
            
            # Glavni delec
            pygame.draw.circle(surface, 
                             (int(r * 255), int(g * 255), int(b * 255), alpha),
                             (int(x), int(y)), int(max(1, size)))
            
            # Dodaj sijaj za boljši videz
            if size > 2:
                highlight_size = max(1, int(size * 0.4))
                highlight_alpha = int(alpha * 0.8)
                highlight_x = x - size * 0.3
                highlight_y = y - size * 0.3
                pygame.draw.circle(surface, 
                                 (255, 255, 255, highlight_alpha),
                                 (int(highlight_x), int(highlight_y)), 
                                 highlight_size)
            
            # Rep delca (za učinek gibanja)
            if i % 3 == 0:  # Samo vsak tretji delec za rep
                tail_length = int(size * 2)
                tail_angle = angle - 0.2  # Rep za delcem
                tail_x = x - math.cos(tail_angle) * tail_length
                tail_y = y - math.sin(tail_angle) * tail_length
                
                for j in range(3):
                    tail_alpha = int(alpha * (1 - j/3) * 0.3)
                    tail_size = max(0.5, size * (1 - j/3))
                    pygame.draw.circle(surface,
                                     (int(r * 255), int(g * 255), int(b * 255), tail_alpha),
                                     (int(tail_x + j * 2), int(tail_y + j * 2)),
                                     int(tail_size))
        
        return surface
    
    def kaleidoscope(self, features, time):
        """Ustvari kalejdoskopski učinek z BOLJŠO KAKOVOSTJO"""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Število sektorjev glede na glasbo
        sectors = 6 + int(features['mid_energy'] * 10)
        
        # Osnovni vzorec - večji in bolj kompleksen
        pattern_size = min(self.width, self.height) // 3
        pattern = pygame.Surface((pattern_size, pattern_size), pygame.SRCALPHA)
        
        # Nariši bolj kompleksen vzorec
        for i in range(self.pattern_elements):
            angle = (i / self.pattern_elements) * math.pi * 2 + time
            radius = pattern_size // 3 * (0.3 + math.sin(time * 1.5 + i * 0.1) * 0.4)
            
            x = pattern_size // 2 + math.cos(angle) * radius
            y = pattern_size // 2 + math.sin(angle) * radius
            
            # Barva z boljšimi prehodi
            hue = (features['brightness'] + i * 0.02 + time * 0.1) % 1.0
            saturation = 0.8
            value = 0.8
            
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            
            size = 2 + features['volume'] * 4
            
            # Glavni krog
            pygame.draw.circle(pattern, 
                             (int(r * 255), int(g * 255), int(b * 255), 200),
                             (int(x), int(y)), int(size))
            
            # Notranji krog za teksturo
            inner_size = max(1, int(size * 0.6))
            inner_color = (min(255, int(r * 255) + 50), 
                          min(255, int(g * 255) + 50), 
                          min(255, int(b * 255) + 50), 
                          150)
            pygame.draw.circle(pattern, inner_color, (int(x), int(y)), inner_size)
            
            # Povezave med točkami za mrežast učinek
            if i > 0 and i % 4 == 0:
                prev_angle = ((i-4) / self.pattern_elements) * math.pi * 2 + time
                prev_radius = pattern_size // 3 * (0.3 + math.sin(time * 1.5 + (i-4) * 0.1) * 0.4)
                prev_x = pattern_size // 2 + math.cos(prev_angle) * prev_radius
                prev_y = pattern_size // 2 + math.sin(prev_angle) * prev_radius
                
                pygame.draw.line(pattern,
                               (int(r * 200), int(g * 200), int(b * 200), 150),
                               (int(prev_x), int(prev_y)),
                               (int(x), int(y)), 2)
        
        # Uporabi kalejdoskopski učinek
        for sector in range(sectors):
            angle = (2 * math.pi / sectors) * sector
            
            # Rotiran in skaliran vzorec
            rotation = math.degrees(angle + time * 6)
            rotated = pygame.transform.rotate(pattern, rotation)
            
            # Dinamično skaliranje
            scale_factor = 0.8 + features['volume'] * 0.4
            scaled_width = int(pattern_size * scale_factor)
            scaled_height = int(pattern_size * scale_factor)
            scaled = pygame.transform.scale(rotated, (scaled_width, scaled_height))
            
            # Pozicioniraj
            blit_rect = scaled.get_rect(center=(self.center_x, self.center_y))
            surface.blit(scaled, blit_rect, special_flags=pygame.BLEND_ADD)
            
            # Dodaj še eno plast z drugačno prosojnostjo
            if features['volume'] > 0.5:
                second_scale = scale_factor * 0.7
                second_scaled = pygame.transform.scale(pattern, 
                                                     (int(pattern_size * second_scale),
                                                      int(pattern_size * second_scale)))
                second_rotated = pygame.transform.rotate(second_scaled, -rotation * 0.5)
                second_rect = second_rotated.get_rect(center=(self.center_x, self.center_y))
                surface.blit(second_rotated, second_rect, special_flags=pygame.BLEND_ADD)
        
        return surface

class OptimizedFractalGenerator:
    """Optimizirana verzija za vnaprejšnje izračunavanje"""
    
    def __init__(self, width, height, quality="high"):
        self.width = width
        self.height = height
        self.quality = quality
        self.fractal_gen = FractalGenerator(width, height, quality)
        
    def generate_fractal_frame(self, features, time):
        """Ustvari en okvir z več fraktali - optimizirano"""
        
        # Izberi glavni fraktal glede na glasbo
        fractal_type = int(features['combined_energy'] * 4) % 4
        
        # Ustvari fraktal
        if fractal_type == 0:
            fractal = self.fractal_gen.mandelbrot(features, time)
        elif fractal_type == 1:
            fractal = self.fractal_gen.julia(features, time)
        elif fractal_type == 2:
            fractal = self.fractal_gen.particle_system(features, time)
        else:
            fractal = self.fractal_gen.kaleidoscope(features, time)
        
        # Dodaj vizualizacijo glasbe
        self.add_simple_audio_visualization(fractal, features)
        
        return fractal
    
    def add_simple_audio_visualization(self, surface, features):
        """Poenostavljena vizualizacija glasbe"""
        width, height = surface.get_size()
        
        # Volume indikator
        vol_size = int(15 + features['volume'] * 25)
        vol_x = width - 25
        vol_y = 25
        
        vol_color = (int(150 + 100 * features['brightness']), 
                    int(80 + 40 * features['volume']), 
                    int(150 + 100 * (1 - features['brightness'])))
        
        pygame.draw.circle(surface, (*vol_color, 180), (vol_x, vol_y), vol_size)
        
        # Beat indikator
        if features['is_beat']:
            pygame.draw.circle(surface, (255, 80, 80, 220),
                             (vol_x, vol_y), vol_size + 4, 2)
            
            # Pulsirajoč učinek
            pulse_size = int(vol_size * (1.5 + math.sin(features['time'] * 20) * 0.3))
            pygame.draw.circle(surface, (255, 100, 100, 80),
                             (vol_x, vol_y), pulse_size, 1)

class PrecomputedFractals:
    """Razred za vnaprej izračunane fraktale - VEČ OKVIRJEV"""
    
    def __init__(self, audio_file, fps=30, width=800, height=600, quality="high"):
        self.audio_file = audio_file
        self.fps = fps
        self.width = width
        self.height = height
        self.quality = quality
        self.fractal_sequence = []
        self.audio_analyzer = AudioAnalyzer(audio_file)
        self.fractal_gen = OptimizedFractalGenerator(width, height, quality)
        
    def precompute_fractals(self, force_recompute=False, max_frames=None):
        """Izračuna vse fraktale vnaprej - VEČ OKVIRJEV"""
        print("=" * 60)
        print("VNAPREJŠNJE IZRAČUNAVANJE FRAKTALOV")
        print(f"Kakovost: {self.quality.upper()}")
        print("=" * 60)
        
        # Preveri cache
        if not force_recompute:
            cache_loaded = self.load_from_cache()
            if cache_loaded:
                return True
        
        # Analiziraj glasbo
        print("Analiziram glasbo...")
        audio_data = self.audio_analyzer.analyze_audio()
        if audio_data is None:
            print("Analiza glasbe ni uspela! Uporabljam testne podatke...")
            duration = 180.0  # 3 minute za test
        else:
            duration = audio_data['duration']
        
        # Določi število okvirjev
        if max_frames is None:
            # Vprašaj uporabnika, koliko sekund želi
            print(f"\nTrajanje glasbe: {duration:.1f}s")
            print(f"FPS: {self.fps}")
            
            # Ponudi možnosti
            print("\nKoliko sekund želite izračunati?")
            print("  1. 30 sekund (hitro testiranje)")
            print("  2. 60 sekund (standardno)")
            print("  3. 120 sekund (dolgo)")
            print(f"  4. Celotno trajanje ({duration:.0f}s)")
            
            choice = input("Izberi možnost (1-4): ").strip()
            if choice == "1":
                max_seconds = 30
            elif choice == "2":
                max_seconds = 60
            elif choice == "3":
                max_seconds = 120
            else:
                max_seconds = duration
            
            total_frames = min(int(max_seconds * self.fps), int(duration * self.fps))
        else:
            total_frames = min(max_frames, int(duration * self.fps))
        
        print(f"\nIzračunavam {total_frames} okvirjev (~{total_frames/self.fps:.1f}s)...")
        print("To lahko traja nekaj minut...")
        
        # Za vsak časovni okvir
        start_time = pygame.time.get_ticks()
        last_update_time = start_time
        
        for frame in range(total_frames):
            time_sec = frame / self.fps
            
            # Pridobi značilnosti glasbe za ta čas
            if audio_data is not None:
                features = self.audio_analyzer.get_audio_features_at_time(time_sec)
            else:
                # Uporabi testne značilnosti
                features = self.audio_analyzer.get_default_features(time_sec)
                features['progress'] = time_sec / duration
            
            # Ustvari fraktal
            fractal_surface = self.fractal_gen.generate_fractal_frame(features, time_sec)
            
            # Shrani
            self.fractal_sequence.append({
                'surface': fractal_surface.copy(),
                'features': features,
                'time': time_sec
            })
            
            # Prikaz napredka
            current_time = pygame.time.get_ticks()
            if frame % 5 == 0 or current_time - last_update_time > 2000:
                last_update_time = current_time
                progress = (frame + 1) / total_frames * 100
                elapsed = (current_time - start_time) / 1000
                eta = (elapsed / (frame + 1)) * (total_frames - frame - 1) if frame > 0 else 0
                
                print(f"  {progress:.1f}% - {frame+1}/{total_frames} - ETA: {eta/60:.1f}min")
        
        # Shrani v cache
        self.save_to_cache()
        
        end_time = pygame.time.get_ticks()
        total_time = (end_time - start_time) / 1000
        
        print(f"\n✓ Izračun končan!")
        print(f"  Skupni čas: {total_time/60:.1f} minut")
        print(f"  Okvirjev: {len(self.fractal_sequence)}")
        print(f"  Povprečje: {total_time/len(self.fractal_sequence)*1000:.1f}ms na okvir")
        
        return True
    
    def save_to_cache(self):
        """Shrani izračunane fraktale v cache"""
        try:
            if not self.fractal_sequence:
                return False
                
            # Shrani kot pickle
            cache_data = {
                'fractal_sequence': self.fractal_sequence,
                'audio_file': self.audio_file,
                'fps': self.fps,
                'width': self.width,
                'height': self.height,
                'quality': self.quality,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('fractal_cache.pkl', 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = os.path.getsize('fractal_cache.pkl') / (1024*1024)
            print(f"Cache shranjen: fractal_cache.pkl ({file_size:.1f} MB)")
            return True
        except Exception as e:
            print(f"Napaka pri shranjevanju cache: {e}")
            return False
    
    def load_from_cache(self, max_frames=None):
        """Naloži izračunane fraktale iz cache"""
        # Najprej preveri .pkl
        if os.path.exists('fractal_cache.pkl'):
            try:
                print("Nalagam iz cache...")
                with open('fractal_cache.pkl', 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.fractal_sequence = cache_data['fractal_sequence']
                
                # Omeji število okvirjev, če je podano
                if max_frames and len(self.fractal_sequence) > max_frames:
                    self.fractal_sequence = self.fractal_sequence[:max_frames]
                
                print(f"✓ Naloženo {len(self.fractal_sequence)} okvirjev iz fractal_cache.pkl")
                return True
            except Exception as e:
                print(f"Napaka pri nalaganju .pkl cache: {e}")
        
        # Če .pkl ne obstaja, preveri .npz
        if os.path.exists('fractal_cache.npz'):
            try:
                print("Nalagam iz starega cache (npz)...")
                data = np.load('fractal_cache.npz', allow_pickle=True)
                
                if 'fractal_sequence' in data:
                    self.fractal_sequence = data['fractal_sequence'].tolist()
                elif 'arrays' in data:
                    arrays = data['arrays']
                    features_list = data['features']
                    times = data['times']
                    
                    self.fractal_sequence = []
                    for i in range(len(arrays)):
                        array = arrays[i]
                        surface = pygame.Surface((self.width, self.height))
                        pygame.surfarray.blit_array(surface, array)
                        self.fractal_sequence.append({
                            'surface': surface,
                            'features': features_list[i].item(),
                            'time': times[i]
                        })
                
                # Omeji število okvirjev
                if max_frames and len(self.fractal_sequence) > max_frames:
                    self.fractal_sequence = self.fractal_sequence[:max_frames]
                
                print(f"✓ Naloženo {len(self.fractal_sequence)} okvirjev iz fractal_cache.npz")
                return True
            except Exception as e:
                print(f"Napaka pri nalaganju .npz cache: {e}")
        
        print("Cache ni najden")
        return False

class MusicFractalPlayer:
    """Glavni razred za predvajanje glasbe z fraktali"""
    
    def __init__(self, audio_file=None, width=800, height=600, quality="high"):
        self.width, self.height = width, height
        self.quality = quality
        
        try:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(f"Fraktali v ritmu glasbe [{quality.upper()}]")
        except Exception as e:
            print(f"Napaka pri ustvarjanju okna: {e}")
            self.screen = None
        
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.running = False
        
        self.audio_file = audio_file
        
        # Za vnaprej izračunane fraktale
        self.precomputed = PrecomputedFractals(audio_file if audio_file else "test", 
                                             fps=30, width=800, height=600, quality=quality)
        
        # Preveri mixer
        self.mixer_available = pygame.mixer.get_init() is not None
        
        # Shranjene fraktale
        self.fractal_sequence = []
        self.current_frame = 0
        
        # Informacije
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Status
        self.music_playing = False
        self.playback_speed = 1.0
        
        # Performančne statistike
        self.frame_times = deque(maxlen=60)
        self.avg_fps = 60
        
    def load_music(self):
        """Naloži glasbo"""
        if not self.mixer_available or not self.audio_file:
            return False
            
        try:
            pygame.mixer.music.load(self.audio_file)
            print(f"Glasba naložena: {os.path.basename(self.audio_file)}")
            return True
        except Exception as e:
            print(f"Napaka pri nalaganju glasbe: {e}")
            return False
    
    def precompute(self, max_frames=None):
        """Izračuna vse fraktale vnaprej"""
        if self.screen is None:
            print("Ustvarjam začasno okno za izračun...")
            try:
                self.screen = pygame.display.set_mode((1, 1), pygame.NOFRAME)
            except:
                pass
        
        success = self.precomputed.precompute_fractals(max_frames=max_frames)
        if success:
            self.fractal_sequence = self.precomputed.fractal_sequence
        return success
    
    def run(self):
        """Zaženi glavno zanko"""
        if not self.fractal_sequence:
            print("Fraktali niso bili izračunani.")
            return
        
        # Naloži glasbo
        music_loaded = self.load_music() if self.audio_file else False
        
        print(f"\n{'='*60}")
        print("PREDVAJANJE FRAKTALOV")
        print(f"{'='*60}")
        print(f"Okvirjev: {len(self.fractal_sequence)}")
        print(f"Trajanje: {len(self.fractal_sequence)/self.precomputed.fps:.1f}s")
        print(f"Kakovost: {self.quality.upper()}")
        
        if music_loaded:
            print("Glasba pripravljena")
        else:
            print("Brez glasbe (samo vizualizacija)")
        
        print("\nKontrolne tipke:")
        print("  ESC: Izhod")
        print("  SPACE: Začni/Ustavi")
        print("  S: Shrani sliko")
        print("  R: Restart")
        print("  +/-: Hitrost")
        print(f"{'-'*60}")
        
        start_time = pygame.time.get_ticks()
        last_frame_time = start_time
        
        print("\nPritisni SPACE za začetek...")
        
        # Glavna zanka
        while self.running:
            #print("I'm in the loop")
            current_time = pygame.time.get_ticks()
            frame_delta = current_time - last_frame_time
            self.frame_times.append(frame_delta)
            last_frame_time = current_time
            
            # Izračunaj FPS
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.avg_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 60
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        if self.music_playing:
                            if music_loaded:
                                pygame.mixer.music.pause()
                            self.music_playing = False
                        else:
                            if music_loaded:
                                pygame.mixer.music.play()
                            self.music_playing = True
                            start_time = pygame.time.get_ticks() - self.current_frame * (1000 / self.precomputed.fps)
                    elif event.key == pygame.K_s:
                        self.save_current_frame()
                    elif event.key == pygame.K_r:
                        self.current_frame = 0
                        start_time = pygame.time.get_ticks()
                        if music_loaded and self.music_playing:
                            pygame.mixer.music.rewind()
                            pygame.mixer.music.play()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.playback_speed = min(3.0, self.playback_speed + 0.1)
                        if music_loaded:
                            pygame.mixer.music.set_volume(min(1.0, 0.8 / self.playback_speed))
                    elif event.key == pygame.K_MINUS:
                        self.playback_speed = max(0.25, self.playback_speed - 0.1)
                        if music_loaded:
                            pygame.mixer.music.set_volume(min(1.0, 0.8 / self.playback_speed))
            
            # Izračunaj trenutni okvir
            if self.music_playing:
                elapsed = (current_time - start_time) * self.playback_speed
                self.current_frame = int(elapsed / (1000 / self.precomputed.fps))
            
            # Preveri meje
            if self.current_frame >= len(self.fractal_sequence):
                self.current_frame = 0
                start_time = current_time
                if music_loaded and self.music_playing:
                    pygame.mixer.music.rewind()
                    pygame.mixer.music.play()
            
            # Pridobi in prikaži trenutni okvir
            if 0 <= self.current_frame < len(self.fractal_sequence):
                frame_data = self.fractal_sequence[self.current_frame]
                fractal_surface = frame_data['surface']
                
                # Skaliraj če je potrebno
                if fractal_surface.get_size() != (self.width, self.height):
                    scaled_fractal = pygame.transform.scale(fractal_surface, (self.width, self.height))
                else:
                    scaled_fractal = fractal_surface
                
                self.screen.blit(scaled_fractal, (0, 0))
                
                # Informacije
                self.display_info(frame_data['features'])
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        # Počisti
        if music_loaded:
            pygame.mixer.music.stop()
    
    def display_info(self, features):
        """Prikaže informacije"""
        # Ozadje
        info_bg = pygame.Surface((250, 140))
        info_bg.set_alpha(180)
        info_bg.fill((10, 10, 20))
        self.screen.blit(info_bg, (10, 10))
        
        # Informacije
        info_lines = [
            f"Čas: {features['time']:.1f}s",
            f"Progress: {features['progress']*100:.0f}%",
            f"FPS: {self.avg_fps:.1f}",
            f"Hitrost: {self.playback_speed:.1f}x",
            f"Okvir: {self.current_frame}/{len(self.fractal_sequence)}"
        ]
        
        y_offset = 15
        for line in info_lines:
            text = self.small_font.render(line, True, (220, 220, 255))
            self.screen.blit(text, (20, y_offset))
            y_offset += 25
        
        # Status
        status = "▶" if self.music_playing else "⏸"
        status_color = (100, 255, 100) if self.music_playing else (255, 200, 100)
        status_text = self.font.render(status, True, status_color)
        self.screen.blit(status_text, (self.width - 50, 20))
    
    def save_current_frame(self):
        """Shrani trenutni okvir kot sliko"""
        if 0 <= self.current_frame < len(self.fractal_sequence):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fractal_{timestamp}_{self.current_frame:06d}.png"
            
            frame_data = self.fractal_sequence[self.current_frame]
            fractal_surface = frame_data['surface']
            
            # Shrani v originalni ločljivosti
            pygame.image.save(fractal_surface, filename)
            print(f"✓ Shranjeno: {filename}")

def main():
    print("=" * 60)
    print("MUSIC FRACTAL GENERATOR - BOLJŠA KAKOVOST")
    print("=" * 60)
    
    # Preveri knjižnice
    try:
        import librosa
        import numpy
        print("Vse knjižnice na voljo")
    except ImportError as e:
        print(f"Manjkajoča knjižnica: {e}")
        print("Namestite s: pip install pygame numpy librosa")
        return
    
    # Poišči WAV datoteke
    wav_files = []
    if os.path.exists("audio"):
        wav_files = [f for f in os.listdir("audio") if f.lower().endswith('.wav')]
    
    if wav_files:
        print("\nNa voljo glasbene datoteke:")
        for i, file in enumerate(wav_files, 1):
            file_path = os.path.join("audio", file)
            size = os.path.getsize(file_path) / (1024*1024)
            print(f"  {i}. {file} ({size:.1f} MB)")
    
    print(f"  {len(wav_files)+1}. Testiranje")
    
    try:
        choice = input(f"\nIzberi glasbo (1-{len(wav_files)+1}): ")
        if choice == str(len(wav_files)+1):
            audio_file = None
            print("Testni način - brez glasbe")
        elif choice.isdigit() and 1 <= int(choice) <= len(wav_files):
            audio_file = os.path.join("audio", wav_files[int(choice)-1])
        else:
            print("Neveljavna izbira!")
            return
    except:
        print("Neveljavna izbira!")
        return
    
    # Kakovost
    print("\nIzberi kakovost:")
    print("  1. Standard (hitreje)")
    print("  2. Visoka (priporočeno)")
    print("  3. Ultra (najlepša, zelo počasno)")
    
    quality_choice = input("Izberi kakovost (1-3): ").strip()
    if quality_choice == "1":
        quality = "low"
    elif quality_choice == "2":
        quality = "high"
    elif quality_choice == "3":
        quality = "ultra"
    else:
        quality = "high"
    
    # Velikost okna
    print("\nIzberi velikost okna:")
    print("  1. 800x600 (hitro)")
    print("  2. 1024x768 (standard)")
    print("  3. 1280x720 (HD)")
    
    size_choice = input("Izberi velikost (1-3): ").strip()
    if size_choice == "1":
        width, height = 800, 600
    elif size_choice == "2":
        width, height = 1024, 768
    elif size_choice == "3":
        width, height = 1280, 720
    else:
        width, height = 1024, 768
    
    player = MusicFractalPlayer(audio_file, width, height, quality)
    
    # Preveri cache
    cache_exists = os.path.exists('fractal_cache.pkl') or os.path.exists('fractal_cache.npz')
    
    print("\n" + "="*60)
    print("Možnosti:")
    if cache_exists:
        print("1. Naloži iz cache (hitro)")
    print("2. Izračunaj 30 sekund (hitro testiranje)")
    print("3. Izračunaj 60 sekund (standardno)")
    print("4. Izračunaj celotno glasbo (dolgo)")
    print("5. Izhod")
    
    choice = input("Izberi možnost: ").strip()
    
    if choice == "1" and cache_exists:
        print("\nNalagam iz cache...")
        # Naloži vse okvirje iz cache
        if player.precomputed.load_from_cache():
            player.fractal_sequence = player.precomputed.fractal_sequence
            print(f"Naloženo {len(player.fractal_sequence)} okvirjev")
            player.running = True
            player.run()
        else:
            print("Napaka pri nalaganju cache. Izračunavam nove...")
            if player.precompute(max_frames=900):  # 30 sekund
                player.running = True
                player.run()
    
    elif choice == "2":
        print("\nIzračunavam 30 sekund fraktalov...")
        if player.precompute(max_frames=900):  # 30 sekund pri 30 FPS
            player.running = True
            player.run()
    
    elif choice == "3":
        print("\nIzračunavam 60 sekund fraktalov...")
        if player.precompute(max_frames=1800):  # 60 sekund pri 30 FPS
            player.running = True
            player.run()
    
    elif choice == "4":
        print("\nIzračunavam celotno glasbo...")
        if player.precompute(max_frames=None):  # Vse okvirje
            player.running = True
            player.run()
    
    else:
        print("Izhod.")

if __name__ == "__main__":
    main()