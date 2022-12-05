#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pretty_midi
import random
import numpy as np
import torch
import pypianoroll
from pypianoroll import Multitrack, Track, StandardTrack
import mido
from mido import MidiFile, tick2second
import copy
import os
import csv
import pydub
import glob
from midi2audio import FluidSynth

n_tracks = 5
latent_dim = 128
n_samples = 4
n_measures = 4
beat_resolution = 4
measure_resolution = 4 * beat_resolution
n_pitches = 72  # number of pitches

programs = [0, 0, 25, 33, 48]
is_drums = [True, False, False, False, False]  # drum indicator for each track
track_names = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']  # name of each track
lowest_pitch = 24
tempo = 100
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)


# In[2]:


class SparkMuse:
    def __init__(self, emotion_csv, highlight_start, highlight_end):
        self.emotion_csv = emotion_csv
        self.emotion_data = {"sec": None, "p": None, "a": None}
        self.set_emotion_data()
        
        self.highlight_start = highlight_start
        self.highlight_end = highlight_end
        self.midi_path = './spark_music_file/'
        if not os.path.isdir(self.midi_path):
            os.makedirs(self.midi_path) 
        self.result_path = self.midi_path + 'result.mid'
        self.intro_path = self.midi_path + 'intro.mid'
        self.main_theme_path = self.midi_path + "main_theme.mid"
        self.outro_path = self.midi_path + "outro.mid"
        self.result_wav = self.midi_path + "result.wav"
        self.result_mp3 = self.midi_path + "result.mp3"

        self.chord_emotion = {"Strong_Positive": ["two"], "Positive": ["six"], "Neutral": [
            "one", "nine", "five"], "Strong_Negative": ["seven"], "Negative": ["three", "four", "eight"]}
        self.chord_data = {"C": [60, 64, 67], "C/E": [64, 67, 72], "D": [62, 66, 69], "E": [64, 68, 71], "F": [
            65, 69, 72], "G": [67, 71, 74], "G/B": [71, 74, 79], "A": [69, 72, 76], "B": [71, 75, 78],
            "Cm": [60, 63, 67], "Dm": [62, 65, 69], "Em": [64, 67, 71], "Em/G": [67, 71, 76],
            "Fm": [65, 68, 72], "Gm": [67, 72, 76], "Am": [69, 72, 76], "Bm": [71, 74, 78]}
        self.chord_progression_data = {"one": ["C", "G/B", "Am", "Em/G", "F", "C/E", "Dm", "G"],
                                       "two": ["C", "Am", "F", "G"], "three": ["F", "G", "Em", "Am", "Dm", "G", "C"],
                                       "four": ["Dm", "G", "C", "Am", "Dm", "Em", "F", "G"], "five": ["F", "Em", "Dm", "G", "C"],
                                       "six": ["C", "G", "Am", "Em", "F", "C", "F", "G"], "seven": ["Am", "G", "F", "Em", "Dm"],
                                       "eight": ["Em", "Am", "F", "G"], "nine": ["C", "Am", "Dm", "G", "C", "Em", "F", "G"]}
        self.melody = {"one": [60, 71, 69, 67, 65, 64, 62, 67], "two": [60, 69, 65, 67], "three": [65, 67, 64, 69, 62, 67, 60],
                       "four": [62, 67, 64, 69, 62, 67, 60], "five": [65, 64, 62, 67, 60], "six": [60, 67, 69, 64, 65, 60, 65, 67],
                       'seven': [69, 67, 65, 64, 62], 'eight': [64, 69, 65, 67], 'nine': [60, 69, 62, 67, 60, 64, 65, 67]}
        self.rhythm_type = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.minor_key_list = [["E", "A", "B"], ["B"], ["B", "E"]]
        self.chords = []
        self.base_notes = []

    def set_emotion_data(self):
        csv_data = []
        sec_list = []
        p_value_list = []
        a_value_list = []
        file = open(self.emotion_csv)
        csv_reader = csv.reader(file)
        for row in csv_reader:
            csv_data.append(row)
        for row in csv_data[1:]:
            sec_list.append(row[1])
            p_value_list.append(row[2])
            a_value_list.append(row[3])
        self.emotion_data["sec"] = sec_list
        self.emotion_data["p"] = p_value_list
        self.emotion_data["a"] = a_value_list
        
        

    def set_chords(self, chord_progression):
        self.chords = []
        for chord in self.chord_progression_data[chord_progression]:
            self.chords.append(self.chord_data[chord])

    def set_base_notes(self, chord_progression):
        self.base_notes = []
        for i in range(len(self.melody[chord_progression])):
            self.base_notes.append(self.melody[chord_progression][i]-12)

    def get_result_path(self):
        return self.result_mp3
    
    def get_emotion_type(self, index):
        emotion_type = 'Neutral'
        p_value = float(self.emotion_data["p"][index])
        a_value = float(self.emotion_data["a"][index])
        if p_value <= 0.1 and p_value >= -0.1:
            emotion_type = 'Neutral'
        elif a_value > 0.5:
            if p_value > 0:
                emotion_type = 'Strong_Positive'
            else:
                emotion_type = 'Strong_Negative'
        else:
            if p_value > 0:
                emotion_type = 'Positive'
            else:
                emotion_type = 'Negative'
        return emotion_type
    
    def get_emotion_max(self, start, end):
        emotion_max = 0
        emotion_max_index = 0
        for i in range(start, end):
            a_value = float(self.emotion_data["a"][i])
            if a_value > emotion_max:
                emotion_max = a_value
                emotion_max_index = i
        return emotion_max_index
    
    def get_emotion_type_chord(self, emotion_type):
        chord = self.chord_emotion[emotion_type]
        emotion_type_chord = random.choice(chord)
        return emotion_type_chord
    
    def chord_fit_time(self, piano_chord, original_time, new_time):
        multiple = new_time/original_time
        for i in range(len(piano_chord.instruments)):
            for note in piano_chord.instruments[i].notes:
                note.start *= multiple
                note.end *= multiple
                time = note.end
        return piano_chord
    
    def generate(self):
        # create pretty midi object for creating chord
        #intro
        piano_chord = pretty_midi.PrettyMIDI()
        time = 0
        start = 0
        end = self.highlight_start//2
        while(start < end):
            if end - start < 10:
                break
            emotion_max_index = self.get_emotion_max(start, start+8)
            emotion_type = self.get_emotion_type(emotion_max_index)
            emotion_type_chord = self.get_emotion_type_chord(emotion_type)
            piano_chord, time = self.create_chord(piano_chord, emotion_type_chord, "0", time)
            start = time
        start = time
        end = self.highlight_start
        while(start < end):
            if end - start < 16:
                break
            emotion_max_index = self.get_emotion_max(start, start+16)
            emotion_type = self.get_emotion_type(emotion_max_index)
            emotion_type_chord = self.get_emotion_type_chord(emotion_type)
            rhythm_type = random.choice(self.rhythm_type)
            piano_chord, time = self.create_chord(piano_chord, emotion_type_chord, rhythm_type, time)
            start = time
        
        piano_chord = self.chord_fit_time(piano_chord, time, end-1)
        piano_chord.write(self.intro_path)
        
        #outro
        piano_chord = pretty_midi.PrettyMIDI()
        
        start = self.highlight_end+1
        end = start + (len(self.emotion_data["sec"])-start)//2
        time = 0
        while(start < end):
            if end - start < 16:
                break
            emotion_max_index = self.get_emotion_max(start, start+16)
            emotion_type = self.get_emotion_type(emotion_max_index)
            emotion_type_chord = self.get_emotion_type_chord(emotion_type)
            rhythm_type = random.choice(self.rhythm_type)
            piano_chord, time = self.create_chord(piano_chord, emotion_type_chord, rhythm_type, time)
            start = time
        start = time
        end = len(self.emotion_data["sec"])
        while(start < end):
            if end - start < 16:
                break
            emotion_max_index = self.get_emotion_max(start, start+16)
            emotion_type = self.get_emotion_type(emotion_max_index)
            emotion_type_chord = self.get_emotion_type_chord(emotion_type)
            rhythm_type = random.choice(self.rhythm_type)
            piano_chord, time = self.create_chord(piano_chord, emotion_type_chord, "0", time)
            start = time
        piano_chord = self.chord_fit_time(piano_chord, time, end-(self.highlight_end+1) -1)
        piano_chord.write(self.outro_path)
        
        #main theme
        emotion_max_index = self.get_emotion_max(self.highlight_start, self.highlight_end)
        emotion_type = self.get_emotion_type(emotion_max_index)
        self.create_main_theme(emotion_type)
        if emotion_type == 'Strong_Negative' or emotion_type == 'Negative':
            self.change_key_to_minor()
        
        # merge three part
        self.cut_music(self.highlight_end - self.highlight_start)
        self.merge_music(self.intro_path, self.main_theme_path)
        self.merge_music(self.result_path, self.outro_path)
        
        self.midi_to_mp3()

        result = self.get_result_path()
        return result

    def create_main_theme(self , emotion_type):
        model_list = []
        emotion_model_dir = ".\\model\\" + emotion_type
        for path in os.listdir(emotion_model_dir):
            filepath = os.path.join(emotion_model_dir, path)
            if os.path.isfile(filepath):
                model_list.append(filepath)
        emotion_model = random.choice(model_list)
        main_theme = MuseGenerator(
            model_path=emotion_model, result_path=self.main_theme_path)

    def cut_music(self, time_length):
        mid = MidiFile(self.main_theme_path)
        mid_cut = copy.deepcopy(mid)
        for i in range(len(mid_cut.tracks)):
            mid_cut.tracks[i].clear()
        for i in range(len(mid.tracks)):
            time = 0
            for msg in mid.tracks[i]:
                time += msg.time
                ab_time = tick2second(time, mid.ticks_per_beat, 600000)
                if ab_time <= time_length:
                    mid_cut.tracks[i].append(msg)
        mid_cut.save(self.main_theme_path)

    def get_etime(self, midi):
        max_etime = 0
        for track in midi.tracks:
            etime = 0
            for msg in track:
                etime += msg.time
            if etime > max_etime:
                max_etime = etime
        return max_etime

    def merge_music(self, front_file_path, next_file_path):
        mid_front = MidiFile(front_file_path)
        mid_next = MidiFile(next_file_path)
        if len(mid_next.tracks) > 2:
            for track in mid_next.tracks:
                for msg in track:
                    if not msg.is_meta:
                        if (msg.type == 'note_on' or msg.type == 'note_off'):
                            msg.time = int(msg.time*1)
        mid_front_etime = self.get_etime(mid_front)
        for track in mid_next.tracks:
            track[0].time = mid_front_etime + 480
            # mid_next append to mid_front
            mid_front.tracks.append(track)
        mid_front.save(self.result_path)
    
    def midi_to_mp3(self):
        #mid to wav
        soundfont = "sound_font\\sound_font1.sf2"
        fs = FluidSynth(sound_font=soundfont)#, sample_rate=22050)
        fs.midi_to_audio(self.result_path, self.result_wav)
        # wav to mp3
        wav_files = glob.glob('./spark_music_file/*.wav')
        for wav_file in wav_files:
            mp3_file = os.path.splitext(wav_file)[0] + '.mp3'
            sound = pydub.AudioSegment.from_wav(wav_file)
            sound.export(mp3_file, format="mp3")
        
    
    def create_chord(self, piano_chord, chord_progression, rhythm_type, stime):
        self.set_chords(chord_progression)
        self.set_base_notes(chord_progression)

        piano_program = pretty_midi.instrument_name_to_program(
            'Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        whole_note_delta = 2
        half_note_delta = whole_note_delta/2
        quarter_note_delta = whole_note_delta/4
        eighth_note_delta = whole_note_delta/8

        if rhythm_type == "0":
            time = stime
            for base_note in self.base_notes:
                note_number = base_note
                note = pretty_midi.Note(
                    velocity=55, pitch=note_number, start=time, end=whole_note_delta)
                piano.notes.append(note)
                time = time + whole_note_delta
            etime = time
            time = stime
            for chord in self.chords:
                for j in range(4):
                    for chord_note in chord:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                    time = time + quarter_note_delta
                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=quarter_note_delta)
                    piano.notes.append(note)

        if rhythm_type == "1":
            time = stime
            for chord in self.chords:
                for j in range(2):
                    for base_note in chord:
                        note_number = base_note-12
                        note = pretty_midi.Note(
                            velocity=48, pitch=note_number, start=time, end=whole_note_delta)
                        piano.notes.append(note)
                    time = time + whole_note_delta

                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=whole_note_delta)
                    piano.notes.append(note)

            etime = time
            time = stime
            for chord in self.chords:
                for j in range(4):
                    for chord_note in chord:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                    time = time + quarter_note_delta
                for chord_note in chord:
                    note_number = chord_note
                    note = pretty_midi.Note(
                        velocity=45, pitch=note_number, start=time, end=whole_note_delta)
                    piano.notes.append(note)
                time = time + whole_note_delta

        if rhythm_type == "2":
            time = stime
            for chord in self.chords:
                for j in range(2):
                    for base_note in chord:
                        note_number = base_note-12
                        note = pretty_midi.Note(
                            velocity=48, pitch=note_number, start=time, end=whole_note_delta)
                        piano.notes.append(note)
                    time = time + whole_note_delta

                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=whole_note_delta)
                    piano.notes.append(note)
            etime = time
            time = stime
            for chord in self.chords:
                for j in range(2):
                    for chord_note in chord[1:]:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                    time = time + quarter_note_delta
                    for chord_note in chord[:1]:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                    time = time + quarter_note_delta

                for chord_note in chord[1:]:
                    note_number = chord_note
                    note = pretty_midi.Note(
                        velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                    piano.notes.append(note)
                time = time + quarter_note_delta
                for chord_note in chord[:1]:
                    note_number = chord_note
                    note = pretty_midi.Note(
                        velocity=45, pitch=note_number, start=time, end=quarter_note_delta*3)
                    piano.notes.append(note)
                time = time + quarter_note_delta*3

        if rhythm_type == "3":
            time = stime
            for chord in self.chords:
                for j in range(2):
                    for base_note in chord:
                        note_number = base_note-12
                        note = pretty_midi.Note(
                            velocity=48, pitch=note_number, start=time, end=whole_note_delta)
                        piano.notes.append(note)
                    time = time + whole_note_delta

                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=whole_note_delta)
                    piano.notes.append(note)
            etime = time
            time = stime
            for chord in self.chords:
                for j in range(2):
                    for chord_note in chord[::2]:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                    time = time + quarter_note_delta
                    for chord_note in chord[1::2]:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                    time = time + quarter_note_delta

                for chord_note in chord[::2]:
                    note_number = chord_note
                    note = pretty_midi.Note(
                        velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                    piano.notes.append(note)
                time = time + quarter_note_delta
                for chord_note in chord[1::2]:
                    note_number = chord_note
                    note = pretty_midi.Note(
                        velocity=45, pitch=note_number, start=time, end=quarter_note_delta*3)
                    piano.notes.append(note)
                time = time + quarter_note_delta*3

        if rhythm_type == "4":
            time = stime
            for chord in self.chords:
                for j in range(3):
                    for base_note in chord:
                        note_number = base_note-12
                        note = pretty_midi.Note(
                            velocity=48, pitch=note_number, start=time, end=whole_note_delta)
                        piano.notes.append(note)
                    time = time + whole_note_delta

                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=whole_note_delta)
                    piano.notes.append(note)
            etime = time
            time = stime
            for chord in self.chords:
                for j in range(4):
                    for chord_note in chord:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                    time = time + quarter_note_delta

                    for chord_note in chord[2:]:
                        note_number = chord_note-12
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                    time = time + quarter_note_delta
                for chord_note in chord:
                    note_number = chord_note
                    note = pretty_midi.Note(
                        velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                    piano.notes.append(note)
                time = time + quarter_note_delta
                for chord_note in chord+[chord[2]-12]:
                    note_number = chord_note
                    note = pretty_midi.Note(
                        velocity=45, pitch=note_number, start=time, end=quarter_note_delta*3)
                    piano.notes.append(note)
                time = time + quarter_note_delta*3

        if rhythm_type == "5":
            time = stime
            for chord in self.chords:
                for j in range(3):
                    for base_note in chord:
                        note_number = base_note-12
                        note = pretty_midi.Note(
                            velocity=48, pitch=note_number, start=time, end=whole_note_delta)
                        piano.notes.append(note)
                    time = time + whole_note_delta

                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=whole_note_delta)
                    piano.notes.append(note)
            etime = time
            time = stime
            for chord in self.chords:
                for j in range(2):
                    for chord_note in chord:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                        time = time + quarter_note_delta
                    for chord_note in chord[1::2]:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                        time = time + quarter_note_delta

                note_number = chord[0]
                note = pretty_midi.Note(
                    velocity=45, pitch=note_number, start=time, end=whole_note_delta)
                piano.notes.append(note)
                time = time + whole_note_delta

        if rhythm_type == "6":
            time = stime
            for chord in self.chords:
                for j in range(2):
                    time_delta = whole_note_delta
                    if j == 1:
                        time_delta = whole_note_delta*2
                    for base_note in chord:
                        note_number = base_note-12
                        note = pretty_midi.Note(
                            velocity=48, pitch=note_number, start=time, end=time_delta)
                        piano.notes.append(note)
                    time = time + time_delta

                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=time_delta)
                    piano.notes.append(note)
            etime = time
            note_order = [0, 2, 1, 2]
            time = stime
            for chord in self.chords:
                for j in range(2):
                    time_delta = quarter_note_delta
                    for k in range(len(note_order)):
                        if j == 1 and k == (len(note_order)-1):
                            time_delta = quarter_note_delta*5
                        note_number = chord[note_order[k]]
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=time_delta)
                        piano.notes.append(note)
                        time = time + time_delta
        if rhythm_type == "7":
            time = stime
            for chord in self.chords:
                for j in range(2):
                    time_delta = whole_note_delta
                    if j == 1:
                        time_delta = whole_note_delta*2
                    for base_note in chord:
                        note_number = base_note-12
                        note = pretty_midi.Note(
                            velocity=48, pitch=note_number, start=time, end=time_delta)
                        piano.notes.append(note)
                    time = time + time_delta

                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=time_delta)
                    piano.notes.append(note)
            etime = time
            note_order = [2, 0, 1, 0]
            time = stime
            for chord in self.chords:
                for j in range(2):
                    time_delta = quarter_note_delta
                    for k in range(len(note_order)):
                        if j == 1 and k == (len(note_order)-1):
                            time_delta = quarter_note_delta*5
                        note_number = chord[note_order[k]]
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=time_delta)
                        piano.notes.append(note)
                        time = time + time_delta

        if rhythm_type == "8":
            time = stime
            for chord in self.chords:
                for j in range(3):
                    for base_note in chord:
                        note_number = base_note-12
                        note = pretty_midi.Note(
                            velocity=48, pitch=note_number, start=time, end=whole_note_delta)
                        piano.notes.append(note)
                    time = time + whole_note_delta

                    note_number = 21
                    note = pretty_midi.Note(
                        velocity=0, pitch=note_number, start=time, end=whole_note_delta)
                    piano.notes.append(note)
            etime = time
            time = stime
            for chord in self.chords:
                for j in range(2):
                    for chord_note in chord:
                        note_number = chord_note
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)

                        if chord_note == chord[2]:
                            note_number = chord[0]
                            note = pretty_midi.Note(
                                velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                            piano.notes.append(note)
                        time = time + quarter_note_delta

                    for chord_note in chord[1::2]:
                        note_number = (chord_note)
                        note = pretty_midi.Note(
                            velocity=45, pitch=note_number, start=time, end=quarter_note_delta)
                        piano.notes.append(note)
                        time = time + quarter_note_delta

                note_number = chord[0]
                note = pretty_midi.Note(
                    velocity=45, pitch=note_number, start=time, end=whole_note_delta)
                piano.notes.append(note)
                time = time + whole_note_delta

        # Add the cello instrument to the PrettyMIDI object
        piano_chord.instruments.append(piano)
        return piano_chord, etime
    def change_key_to_minor(self):
        midi = pretty_midi.PrettyMIDI(self.main_theme_path)
        for track in midi.instruments:
            for note in track.notes:
                pitch = pretty_midi.note_number_to_name(note.pitch)
                if pitch[0] in self.minor_key_list[0] and pitch[1]!='#':
                    note.pitch = note.pitch -1
        midi.write(self.main_theme_path)


# In[3]:


class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)
class Generator(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self):
        super().__init__()
        self.transconv0 = GeneraterBlock(latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = torch.nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(n_tracks)
        ])
        self.transconv5 = torch.nn.ModuleList([
            GeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])

    def forward(self, x):
        x = x.view(-1, latent_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = [transconv(x) for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)
        x = x.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return x
class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.layernorm(x)
        return torch.nn.functional.leaky_relu(x)
class Discriminator(torch.nn.Module):
    """A convolutional neural network (CNN) based discriminator. The
    discriminator takes as input either a real sample (in the training data) or
    a fake sample (generated by the generator) and outputs a scalar indicating
    its authentity.
    """
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks)
        ])
        self.conv1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 16, (1, 4, 1), (1, 4, 1)) for _ in range(n_tracks)
        ])
        self.conv2 = DiscriminatorBlock(16 * n_tracks, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = DiscriminatorBlock(64, 64, (1, 1, 4), (1, 1, 4))
        self.conv4 = DiscriminatorBlock(64, 128, (1, 4, 1), (1, 4, 1))
        self.conv5 = DiscriminatorBlock(128, 128, (2, 1, 1), (1, 1, 1))
        self.conv6 = DiscriminatorBlock(128, 256, (3, 1, 1), (3, 1, 1))
        self.dense = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, n_tracks, n_measures, measure_resolution, n_pitches)
        x = [conv(x[:, [i]]) for i, conv in enumerate(self.conv0)]
        x = torch.cat([conv(x_) for x_, conv in zip(x, self.conv1)], 1)
        x = self.conv2(x)
        x = self.conv3(x)          
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 256)
        x = self.dense(x)
        return x


# In[4]:


class MuseGenerator:
    def __init__(self, model_path,result_path):
        self.model = self.load_model(model_path)
        self.discriminator = None
        self.generator = None
        self.d_optimizer = None
        self.g_optimizer = None
        self.load_model(model_path)
        
        self.result_path = result_path#"midi_file/music.mid"
        self.generate_music()
        self.change_music_velocity()
        # self.cut_music(10)
        
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        # Create neural networks
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.generator.load_state_dict(checkpoint['Generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['Discriminator_state_dict'])
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=0.001, betas=(0.5, 0.9))
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        
    def generate_music(self):
        sample_latent = torch.randn(n_samples, latent_dim)
        # Transfer the neural nets and samples to GPU
        if torch.cuda.is_available():
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()
            sample_latent = sample_latent.cuda()

        sample = self.generator(sample_latent).cpu().detach().numpy()
        sample = sample.transpose(1, 0, 2, 3).reshape(n_tracks, -1, n_pitches)
        tracks = []
        for idx, (program, is_drum, track_name) in enumerate(
            zip(programs, is_drums, track_names)
        ):
            pianoroll = np.pad(
                sample[idx] > 0.5,
                ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
            )
            tracks.append(
                StandardTrack(
                    name=track_name,
                    program=program,
                    is_drum=is_drum,
                    pianoroll=pianoroll
                )
            )
        m = Multitrack(
            tracks=tracks,
            tempo=tempo_array,
            resolution=beat_resolution
        )
        pypianoroll.write(self.result_path,m)
        
    def change_music_velocity(self):
        instrument_list=['Piano', 'Drums', 'Guitar', 'Bass']
        mid = mido.MidiFile(self.result_path)
        for i, track in enumerate(mid.tracks):
            # 如果不是在list裡面不調整音量
            if track.name not in instrument_list:
                continue
            for msg in track:
                if not msg.is_meta:
                    if (msg.type == 'note_on' and msg.velocity != 0):
                        msg.velocity = 64
        # save
        mid.save(self.result_path)


# In[5]:


"""# How to use
highlight_start_time = 60
highlight_end_time = 90
emotion = "fin.csv"
muse = SparkMuse(emotion, highlight_start_time, highlight_end_time)
result_music = muse.generate()"""

