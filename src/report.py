import io
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from src.utils import (
    get_batsman_kpis, 
    get_strike_rate_by_phase, 
    get_strike_rate_by_bowler_style,
    get_dismissals_by_bowler_style,
    get_bowler_kpis,
    get_bowler_economy_by_phase,
    get_bowler_wickets_by_phase,
    get_bowler_average_by_batsman_type
)

def generate_tactical_pdf(player_name, player_type="Batsman"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom Styles
    alert_style = ParagraphStyle(
        'Alert', 
        parent=styles['Normal'],
        textColor=colors.firebrick,
        fontName='Helvetica-Bold',
        fontSize=11,
        spaceAfter=12
    )
    
    elements = []
    
    # Header
    elements.append(Paragraph(f"DeepSequence-T20I: Tactical Plan-of-Attack", title_style))
    elements.append(Paragraph(f"Target Profile: {player_name} ({player_type})", subtitle_style))
    elements.append(Spacer(1, 12))
    
    if player_type == "Batsman":
        # 1. KPIs
        total_runs, balls_faced, strike_rate, times_out = get_batsman_kpis(player_name)
        elements.append(Paragraph("1. True Lifetime KPIs", subtitle_style))
        
        kpi_data = [
            ["Total Runs", "Balls Faced", "Strike Rate", "Times Out"],
            [str(total_runs), str(balls_faced), str(strike_rate), str(times_out)]
        ]
        t = Table(kpi_data, colWidths=[120, 120, 120, 120])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.steelblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 10),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))
        
        # 2. Phase Analysis
        elements.append(Paragraph("2. Match Phase Analytics", subtitle_style))
        df_sr_phase = get_strike_rate_by_phase(player_name)
        
        phase_data = [["Match Phase", "Strike Rate"]]
        for index, row in df_sr_phase.iterrows():
            phase = row['Phase']
            sr = row['Strike Rate']
            phase_data.append([phase, str(sr)])
            
        t_phase = Table(phase_data, colWidths=[240, 240])
        t_phase.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.steelblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(t_phase)
        elements.append(Spacer(1, 20))
        
        # 3. Bowler Style Analysis
        elements.append(Paragraph("3. Vulnerability vs Bowler Styles", subtitle_style))
        df_sr_style = get_strike_rate_by_bowler_style(player_name)
        df_out_style = get_dismissals_by_bowler_style(player_name)
        
        style_data = [["Bowler Style", "Strike Rate", "Dismissals"]]
        out_col = 'Dismissals' if 'Dismissals' in df_out_style.columns else 'dismissals'
        for index, row in df_sr_style.iterrows():
            style = row['Bowler Sub-Style']
            clean_style = style.replace("(Kaggle Dataset)", "").replace("(kaggle dataset)", "").strip()
            sr = row['Strike Rate']
            outs = df_out_style[df_out_style['Bowler Sub-Style'] == style][out_col].values
            outs_val = str(outs[0]) if len(outs) > 0 else "0"
            style_data.append([clean_style, str(sr), outs_val])
            
        t_style = Table(style_data, colWidths=[160, 160, 160])
        t_style.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.steelblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(t_style)
        elements.append(Spacer(1, 20))
        
        # 4. NLP Vulnerability Matrix
        elements.append(Paragraph("4. NLP Machine Learning Insights (Hugging Face Dataset)", subtitle_style))
        try:
            import pandas as pd
            df_hf = pd.read_csv("data/processed/hf_commentary_labels.csv")
            last_name = player_name.split()[-1]
            
            # Find all deliveries mentioning this player
            player_rows = df_hf[df_hf['text'].str.contains(last_name, case=False, na=False)]
            
            if not player_rows.empty:
                stats = {'wickets': 0, 'dots': 0, 'balls': 0, 'lines': {}, 'lengths': {}, 'shots': {}}
                
                for _, row in player_rows.iterrows():
                    text = str(row['text']).lower()
                    line = str(row['line'])
                    length = str(row['length'])
                    shot = str(row['shot'])
                    
                    # Heuristic outcome parsing
                    is_vuln = 0
                    if 'out' in text or 'caught' in text or 'bowled' in text or 'lbw' in text or 'dismissal' in text:
                        stats['wickets'] += 1
                        is_vuln = 1
                    elif 'dot' in text or 'no run' in text:
                        stats['dots'] += 1
                        is_vuln = 1
                        
                    stats['balls'] += 1
                    
                    if line != 'Unknown' and line != 'nan':
                        if line not in stats['lines']: stats['lines'][line] = {'faced': 0, 'vuln': 0}
                        stats['lines'][line]['faced'] += 1
                        stats['lines'][line]['vuln'] += is_vuln
                        
                    if length != 'Unknown' and length != 'nan':
                        if length not in stats['lengths']: stats['lengths'][length] = {'faced': 0, 'vuln': 0}
                        stats['lengths'][length]['faced'] += 1
                        stats['lengths'][length]['vuln'] += is_vuln
                        
                    if shot != 'Unknown' and shot != 'nan':
                        if shot not in stats['shots']: stats['shots'][shot] = {'faced': 0, 'vuln': 0}
                        stats['shots'][shot]['faced'] += 1
                        stats['shots'][shot]['vuln'] += is_vuln
                        
                # Calculate worst mechanics
                worst_line = max(stats['lines'].keys(), key=lambda k: stats['lines'][k]['vuln'] / max(1, stats['lines'][k]['faced'])) if stats['lines'] else 'Unknown'
                worst_len = max(stats['lengths'].keys(), key=lambda k: stats['lengths'][k]['vuln'] / max(1, stats['lengths'][k]['faced'])) if stats['lengths'] else 'Unknown'
                worst_shot = max(stats['shots'].keys(), key=lambda k: stats['shots'][k]['vuln'] / max(1, stats['shots'][k]['faced'])) if stats['shots'] else 'Unknown'
                
                alert_text = f"CRITICAL VULNERABILITY DETECTED: Historical NLP commentary analysis from {stats['balls']} textual deliveries indicates {player_name} is highly susceptible to {worst_len.upper()} deliveries on the {worst_line.upper()} line, especially when attempting the {worst_shot.upper()} shot."
                elements.append(Paragraph(alert_text, alert_style))
                
                # Draw the HF Vulnerability Breakdown Table
                elements.append(Spacer(1, 10))
                hf_table_data = [["Delivery Mechanic", "Balls Faced", "Vulnerability (Wickets/Dots)", "Risk Score %"]]
                
                for k, v in stats['lines'].items(): 
                    pct = (v['vuln'] / max(1, v['faced'])) * 100
                    hf_table_data.append([f"Line: {k}", str(v['faced']), str(v['vuln']), f"{pct:.1f}%"])
                for k, v in stats['lengths'].items(): 
                    pct = (v['vuln'] / max(1, v['faced'])) * 100
                    hf_table_data.append([f"Length: {k}", str(v['faced']), str(v['vuln']), f"{pct:.1f}%"])
                for k, v in stats['shots'].items(): 
                    pct = (v['vuln'] / max(1, v['faced'])) * 100
                    hf_table_data.append([f"Shot: {k}", str(v['faced']), str(v['vuln']), f"{pct:.1f}%"])
                    
                t_hf = Table(hf_table_data, colWidths=[140, 80, 160, 100])
                t_hf.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.darkred),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(t_hf)
                
            else:
                elements.append(Paragraph(f"No specific NLP vulnerabilities found for {player_name} in the Hugging Face dataset.", normal_style))
        except Exception as e:
            elements.append(Paragraph(f"NLP Data not available: {e}", normal_style))
            
    else:
        # Bowler Profile
        wickets, runs, economy, average, sr = get_bowler_kpis(player_name)
        elements.append(Paragraph("1. True Lifetime KPIs", subtitle_style))
        
        kpi_data = [
            ["Wickets", "Economy", "Average", "Strike Rate"],
            [str(wickets), str(economy), str(average), str(sr)]
        ]
        t = Table(kpi_data, colWidths=[120, 120, 120, 120])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.steelblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 10),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # Bowler Phase Analysis
        elements.append(Paragraph("2. Match Phase Analytics", subtitle_style))
        df_econ_phase = get_bowler_economy_by_phase(player_name)
        df_wkt_phase = get_bowler_wickets_by_phase(player_name)
        
        phase_data = [["Match Phase", "Economy", "Wickets"]]
        for index, row in df_econ_phase.iterrows():
            phase = row['Phase']
            econ = row['Economy']
            wkts = df_wkt_phase[df_wkt_phase['Phase'] == phase]['Wickets'].values
            wkts_val = str(wkts[0]) if len(wkts) > 0 else "0"
            phase_data.append([phase, str(econ), wkts_val])
            
        t_phase = Table(phase_data, colWidths=[160, 160, 160])
        t_phase.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.steelblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(t_phase)
        elements.append(Spacer(1, 20))

        # 3. Batsman Type Analysis
        elements.append(Paragraph("3. Analytics vs Batsman Styles", subtitle_style))
        df_bat_type = get_bowler_average_by_batsman_type(player_name)
        
        bat_data = [["Batsman Type", "Economy", "Strike Rate", "Average", "Wickets"]]
        for index, row in df_bat_type.iterrows():
            bat_type = row['Batsman Type']
            clean_bat_type = bat_type.replace("(Kaggle Dataset)", "").replace("(kaggle dataset)", "").strip()
            b_econ = row['Economy']
            b_sr = row['Strike Rate']
            b_avg = row['Average']
            b_wkts = row['Wickets']
            bat_data.append([clean_bat_type, str(b_econ), str(b_sr), str(b_avg), str(b_wkts)])
            
        t_bat = Table(bat_data, colWidths=[100, 70, 70, 70, 70])
        t_bat.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.steelblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(t_bat)
        elements.append(Spacer(1, 20))

        # 4. NLP Lethality Matrix
        elements.append(Paragraph("4. NLP Machine Learning Insights", subtitle_style))
        try:
            df_str = pd.read_csv("data/processed/global_bowler_strengths.csv")
            player_str = df_str[df_str['Player'] == player_name]
            if not player_str.empty:
                p_line = player_str.iloc[0]['Primary_Strength_Line']
                p_length = player_str.iloc[0]['Primary_Strength_Length']
                
                alert_text = f"CRITICAL LETHALITY: Historical NLP commentary analysis indicates {player_name} is most lethal with {p_length.upper()} deliveries on the {p_line.upper()} line."
                elements.append(Paragraph(alert_text, alert_style))
                
                # Bowling Plan Summary
                plan_text = f"RECOMMENDED USAGE: Deploy {player_name} heavily when a batsman is known to be vulnerable to {p_length} lengths on the {p_line}."
                elements.append(Paragraph(plan_text, normal_style))
            else:
                elements.append(Paragraph("No specific NLP strengths found for this bowler.", normal_style))
        except Exception as e:
            elements.append(Paragraph("NLP Data not available.", normal_style))
        
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_presentation_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    normal_style.spaceAfter = 12
    normal_style.fontSize = 11
    
    elements = []
    
    # Title Slide
    elements.append(Spacer(1, 100))
    elements.append(Paragraph("DeepSequence-T20I", title_style))
    elements.append(Paragraph("Contextual Batsman Vulnerability & Strategy Engine", subtitle_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("CSE299.13: Junior Design Project", ParagraphStyle('center', parent=normal_style, alignment=1)))
    elements.append(Paragraph("S M Tazbid Siddiqui & Mostofa Morshed", ParagraphStyle('center', parent=normal_style, alignment=1)))
    elements.append(Spacer(1, 150))
    
    # Problem Statement
    elements.append(Paragraph("1. Problem Statement", subtitle_style))
    elements.append(Paragraph("In T20Is, tactical demands switch rapidly between overs. Traditional flat averages fail to inform live gameplay strategies. Current sports teams struggle with the fallacy of aggregated metrics—overlooking how a tactical sequence of variations changes a batsman's shot risk.", normal_style))
    
    # Solution
    elements.append(Paragraph("2. DeepSequence-T20I Solution", subtitle_style))
    elements.append(Paragraph("Our platform shifts cricket analytics from historical lookups to predictive sequence mapping. It isolates tactical boundaries, maps physical ball trajectories via NLP from text feeds, and trains sequential deep learning (LSTM) architectures to uncover situational weaknesses.", normal_style))
    
    # Architecture
    elements.append(Paragraph("3. Technical Architecture", subtitle_style))
    elements.append(Paragraph("• <b>NLP Parser:</b> Regex indexing engine transforming unstructured commentary into explicit physical variables (Line, Length, Shot).", normal_style))
    elements.append(Paragraph("• <b>Deep Learning:</b> PyTorch LSTM network processing rolling 6-delivery windows (52D Feature Vectors) to predict fatal errors.", normal_style))
    elements.append(Paragraph("• <b>Backend:</b> SQLite Relational Database caching processed Cricsheet logs.", normal_style))
    elements.append(Paragraph("• <b>Frontend:</b> Interactive Streamlit dashboard for real-time tactical simulation and PDF Cheat Sheet generation.", normal_style))
    
    # Results & Economic Value
    elements.append(Paragraph("4. Model Evaluation & Economic Value", subtitle_style))
    elements.append(Paragraph("The focal loss model isolates critical dismissal sequences with an F1-Score of 80% (vs baseline 45%). Economically, this system saves performance analysts over 14 hours of manual video/text scrubbing per match, providing instant, data-driven 'Plan-of-Attack' cheat sheets for franchise setups.", normal_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer
