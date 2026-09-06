import numpy as np
from sklearn.preprocessing import OneHotEncoder

try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    import traceback
    print(f"CRITICAL PYTORCH IMPORT ERROR (features): {e}")
    traceback.print_exc()
    TORCH_AVAILABLE = False

class SequencePreprocessor:
    def __init__(self):
        # We will use simple one-hot encoding for the categorical features
        self.length_categories = ["Yorker", "Full", "Slot", "Good Length", "Short"]
        self.phase_categories = ["Powerplay", "Middle Overs", "Death Overs"]
        self.style_categories = ["Pace", "Off-spin", "Leg-spin"]
        self.line_categories = ["Outside Off", "Off Stump", "Middle Stump", "Leg Stump", "Down Leg", "Unknown"]
        self.shot_categories = ["Defend", "Drive", "Pull", "Cut", "Flick", "Sweep", "Unknown"]
        
        self.encoder_len = OneHotEncoder(categories=[self.length_categories], sparse_output=False, handle_unknown='ignore')
        self.encoder_phase = OneHotEncoder(categories=[self.phase_categories], sparse_output=False, handle_unknown='ignore')
        self.encoder_style = OneHotEncoder(categories=[self.style_categories], sparse_output=False, handle_unknown='ignore')
        self.encoder_line = OneHotEncoder(categories=[self.line_categories], sparse_output=False, handle_unknown='ignore')
        self.encoder_shot = OneHotEncoder(categories=[self.shot_categories], sparse_output=False, handle_unknown='ignore')
        
        # Fit once to initialize
        self.encoder_len.fit([["Yorker"], ["Full"], ["Slot"], ["Good Length"], ["Short"]])
        self.encoder_phase.fit([["Powerplay"], ["Middle Overs"], ["Death Overs"]])
        self.encoder_style.fit([["Pace"], ["Off-spin"], ["Leg-spin"]])
        self.encoder_line.fit([["Outside Off"], ["Off Stump"], ["Middle Stump"], ["Leg Stump"], ["Down Leg"], ["Unknown"]])
        self.encoder_shot.fit([["Defend"], ["Drive"], ["Pull"], ["Cut"], ["Flick"], ["Sweep"], ["Unknown"]])
        
    def encode_categorical(self, value, encoder):
        return encoder.transform([[value]])[0].tolist()

    def preprocess_sequence(self, sequence_data, phase, style, norm_sr, dismissal_rate, 
                            bowler_phase_econ, bowler_phase_wickets, bowler_type_avg, bowler_career_wickets, 
                            bowler_career_econ, bowler_career_avg, bowler_career_sr,
                            batsman_avg_vs_style, bowler_econ_vs_style, bowler_sr_vs_style, bowler_wkts_vs_style,
                            sb_career_wickets, sb_career_econ, sb_career_avg, sb_career_sr,
                            str_runs=0, str_balls=0, nstr_runs=0, nstr_balls=0, 
                            tb_runs=0, tb_wkts=0, sb_runs=0, sb_wkts=0,
                            bat_career_avg=25.0, bat_career_sr=120.0,
                            partnership_strength=0.5, crr=7.5, wickets_fallen=0, rrr=7.5):
        """
        Converts a list of dicts (deliveries) into a PyTorch-ready tensor.
        Extracts categorical features and scales continuous variables.
        
        Args:
            sequence_data: List of dicts `[{'run': 1, 'length': 'Short', 'line': 'Off Stump', 'shot': 'Defend'}, ...]`
            phase: Current Match Phase (e.g., 'Powerplay')
            style: Current Bowler Style (e.g., 'Pace')
            norm_sr: Scaled batsman strike rate (historical)
            dismissal_rate: Scaled batsman dismissal rate (historical)
            bowler_phase_econ: Bowler's economy in the current phase
            bowler_phase_wickets: Bowler's wickets in the current phase
            bowler_type_avg: Bowler's average against the current batsman type
            bowler_career_wickets: Bowler's total career wickets
            bowler_career_econ: Bowler's total career economy
            bowler_career_avg: Bowler's total career average
            bowler_career_sr: Bowler's total career strike rate
            batsman_avg_vs_style: Batsman's average against this bowler's style
            bowler_econ_vs_style: Bowler's economy against this batsman's style
            bowler_sr_vs_style: Bowler's strike rate against this batsman's style
            bowler_wkts_vs_style: Bowler's total wickets against this batsman's style
            sb_career_wickets: Supporting Bowler's career wickets
            sb_career_econ: Supporting Bowler's career economy
            sb_career_avg: Supporting Bowler's career average
            sb_career_sr: Supporting Bowler's career strike rate
            str_runs: Striker's current match runs
            str_balls: Striker's current match balls faced
            nstr_runs: Non-Striker's current match runs
            nstr_balls: Non-Striker's current match balls faced
            tb_runs: Target Bowler's current match runs conceded
            tb_wkts: Target Bowler's current match wickets taken
            sb_runs: Supporting Bowler's current match runs conceded
            sb_wkts: Supporting Bowler's current match wickets taken
            
        Returns:
            torch.Tensor of shape (1, seq_length, 50)
        """
        # Normalization constraints
        norm_b_phase_econ = min(float(bowler_phase_econ) / 15.0, 1.0)
        norm_b_phase_wkts = min(float(bowler_phase_wickets) / 50.0, 1.0)
        
        # Handle 'N/A' or missing averages
        try:
            b_avg_float = float(bowler_type_avg)
            norm_b_type_avg = min(b_avg_float / 50.0, 1.0)
        except ValueError:
            norm_b_type_avg = 0.5 # Default middle ground for unknown
            
        norm_b_wkts = min(float(bowler_career_wickets) / 200.0, 1.0)
        norm_b_career_econ = min(float(bowler_career_econ) / 12.0, 1.0)
        
        try:
            c_avg_float = float(bowler_career_avg)
            norm_b_career_avg = min(c_avg_float / 50.0, 1.0)
        except ValueError:
            norm_b_career_avg = 0.5
            
        try:
            c_sr_float = float(bowler_career_sr)
            norm_b_career_sr = min(c_sr_float / 30.0, 1.0)
        except ValueError:
            norm_b_career_sr = 0.5
            
        # Normalization for the 4 new specific dimensions
        try:
            bat_avg_float = float(batsman_avg_vs_style)
            norm_bat_avg_vs_style = min(bat_avg_float / 60.0, 1.0)
        except ValueError:
            norm_bat_avg_vs_style = 0.5
            
        norm_b_econ_vs_style = min(float(bowler_econ_vs_style) / 15.0, 1.0)
        
        try:
            b_sr_float = float(bowler_sr_vs_style)
            norm_b_sr_vs_style = min(b_sr_float / 30.0, 1.0)
        except ValueError:
            norm_b_sr_vs_style = 0.5
            
        norm_b_wkts_vs_style = min(float(bowler_wkts_vs_style) / 100.0, 1.0)
            
        # Scale Live Match Stats
        n_str_runs = min(str_runs / 150.0, 1.0)
        n_str_balls = min(str_balls / 100.0, 1.0)
        n_nstr_runs = min(nstr_runs / 150.0, 1.0)
        n_nstr_balls = min(nstr_balls / 100.0, 1.0)
        n_tb_runs = min(tb_runs / 80.0, 1.0)
        n_tb_wkts = min(tb_wkts / 10.0, 1.0)
        n_sb_runs = min(sb_runs / 80.0, 1.0)
        n_sb_wkts = min(sb_wkts / 10.0, 1.0)
        
        # Scale Supporting Bowler Career KPIs
        norm_sb_wkts = min(sb_career_wickets / 300.0, 1.0)
        norm_sb_econ = min(sb_career_econ / 15.0, 1.0)
        try:
            norm_sb_avg = min(float(sb_career_avg) / 60.0, 1.0)
        except ValueError:
            norm_sb_avg = 0.5
        try:
            norm_sb_sr = min(float(sb_career_sr) / 40.0, 1.0)
        except ValueError:
            norm_sb_sr = 0.5
            
        # Scale true lifetime Batsman KPIs
        norm_bat_career_avg = min(bat_career_avg / 60.0, 1.0)
        norm_bat_career_sr = min(bat_career_sr / 200.0, 1.0)
            
        # One-hot context vectors
        phase_vec = self.encode_categorical(phase, self.encoder_phase)
        style_vec = self.encode_categorical(style, self.encoder_style)
        
        # Sequence construction
        sequence_vectors = []
        for delivery in sequence_data:
            # Scale Run [0, 6] -> [0, 1]
            run_scaled = min(delivery.get('run', 0) / 6.0, 1.0)
            
            # Encode Delivery Mechanics
            len_vec = self.encode_categorical(delivery.get('length', 'Unknown'), self.encoder_len)
            line_vec = self.encode_categorical(delivery.get('line', 'Unknown'), self.encoder_line)
            shot_vec = self.encode_categorical(delivery.get('shot', 'Unknown'), self.encoder_shot)
            
            # Scale Live Match States (4 new variables)
            norm_crr = min(max(crr / 20.0, 0.0), 1.0)
            norm_wkts = min(wickets_fallen / 10.0, 1.0)
            norm_rrr = min(max(rrr / 24.0, 0.0), 1.0)
            norm_ps = min(max(partnership_strength, 0.0), 1.0)
            
            # Build the 56-Dimensional delivery vector
            vector = [run_scaled] + len_vec + line_vec + shot_vec + phase_vec + style_vec + [
                norm_sr, dismissal_rate, 
                norm_b_phase_econ, norm_b_type_avg, 
                norm_b_wkts, norm_b_career_econ, norm_b_career_avg, norm_b_career_sr,
                norm_bat_avg_vs_style, norm_b_econ_vs_style, norm_b_sr_vs_style, norm_b_wkts_vs_style,
                norm_b_phase_wkts,
                n_str_runs, n_str_balls, n_nstr_runs, n_nstr_balls,
                n_tb_runs, n_tb_wkts, n_sb_runs, n_sb_wkts,
                norm_sb_wkts, norm_sb_econ, norm_sb_avg, norm_sb_sr,
                norm_bat_career_avg, norm_bat_career_sr,
                norm_ps, norm_crr, norm_wkts, norm_rrr
            ]
            sequence_vectors.append(vector)
            
        # Convert to numpy array then tensor
        tensor_data = np.array([sequence_vectors], dtype=np.float32)
        
        if TORCH_AVAILABLE:
            return torch.tensor(tensor_data)
        else:
            return tensor_data
