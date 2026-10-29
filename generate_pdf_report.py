import os
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_charts():
    os.makedirs("pdf_assets", exist_ok=True)
    
    # Chart 1: Few-Shot F1 Progression
    plt.figure(figsize=(6.5, 3.2), dpi=300)
    shots = [0, 10, 50, 100, 200, 500]
    te_f1 = [0.4658, 0.5423, 0.6475, 0.7006, 0.7265, 0.7668]
    kb_shots = [0, 10, 50, 100, 273, 4859]
    kb_f1 = [0.0050, 0.0000, 0.3850, 0.5275, 0.6977, 0.7482]
    
    plt.plot(range(len(shots)), te_f1, marker='o', color='#1E88E5', linewidth=2.5, label='Telugu Target (te)')
    plt.plot(range(len(kb_shots)), kb_f1, marker='s', color='#D81B60', linewidth=2.5, linestyle='--', label='Kokborok Target (trp)')
    
    plt.xticks(range(6), ['0-Shot', '10-Shot', '50-Shot', '100-Shot', '200/273-Shot', '500/4859-Shot'], fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel('Shot Budget / Training Scale', fontsize=10, fontweight='bold')
    plt.ylabel('seqeval F1 Score', fontsize=10, fontweight='bold')
    plt.title('Few-Shot seqeval F1 Performance Curve Across Target Languages', fontsize=11, fontweight='bold', pad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    chart1_path = "pdf_assets/few_shot_curve.png"
    plt.savefig(chart1_path)
    plt.close()
    
    # Chart 2: Parameter Budget Comparison (Log Scale)
    plt.figure(figsize=(6.5, 3.0), dpi=300)
    models = ['IndicNER Baseline', 'XLM-R Base (FT)', 'mBERT Base (FT)', 'Proposed LoRA-r8']
    params = [278.0, 277.45, 177.85, 1.332] # in millions
    colors_list = ['#757575', '#42A5F5', '#AB47BC', '#66BB6A']
    
    bars = plt.barh(models, params, color=colors_list, height=0.55)
    plt.xscale('log')
    plt.xlabel('Trainable Parameters (Millions - Log Scale)', fontsize=10, fontweight='bold')
    plt.title('Trainable Parameter Budget Comparison (Log Scale)', fontsize=11, fontweight='bold', pad=10)
    
    for bar in bars:
        width = bar.get_width()
        val_str = f"{width:.2f}M" if width >= 1 else f"{width*1000:.0f}K ({width/277.45*100:.3f}%)"
        plt.text(width * 1.15, bar.get_y() + bar.get_height()/2, val_str,
                 va='center', ha='left', fontsize=8.5, fontweight='bold')
                 
    plt.xlim(0.5, 600)
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    chart2_path = "pdf_assets/param_comparison.png"
    plt.savefig(chart2_path)
    plt.close()
    
    # Chart 3: Validation Ablation Sweep F1 Gain
    plt.figure(figsize=(6.5, 2.8), dpi=300)
    ablation_names = ['Base LoRA-r8', '+ Soft Prompt (P=20)', '+ Multilingual Warmup', '+ Target Module Expand', '+ 10% Warmup Ratio']
    val_f1s = [0.7215, 0.7289, 0.7584, 0.7692, 0.7240]
    bar_colors = ['#9E9E9E', '#80CBC4', '#26A69A', '#2E7D32', '#B0BEC5']
    
    bars3 = plt.bar(range(len(ablation_names)), val_f1s, color=bar_colors, width=0.5)
    plt.ylim(0.65, 0.80)
    plt.ylabel('Validation Micro F1', fontsize=10, fontweight='bold')
    plt.title('Prompt 2 Validation Ablation Sweep (F1 Gain Analysis)', fontsize=11, fontweight='bold', pad=10)
    plt.xticks(range(len(ablation_names)), ablation_names, rotation=15, ha='right', fontsize=8.5)
    
    for bar in bars3:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    chart3_path = "pdf_assets/ablation_sweep.png"
    plt.savefig(chart3_path)
    plt.close()
    
    return chart1_path, chart2_path, chart3_path

def build_pdf():
    pdf_path = "PROJECT_DOCUMENTATION.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    c1, c2, c3 = generate_charts()
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#1A237E") # Deep Navy
    SECONDARY = colors.HexColor("#0D47A1") # Medium Navy
    ACCENT = colors.HexColor("#00838F") # Cyan
    DARK_TEXT = colors.HexColor("#212121")
    LIGHT_BG = colors.HexColor("#F5F5F5")
    BORDER_COLOR = colors.HexColor("#E0E0E0")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        alignment=1,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=SECONDARY,
        alignment=1,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=DARK_TEXT,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=DARK_TEXT,
        leftIndent=12,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=PRIMARY,
        backColor=colors.HexColor("#E8EAF6"),
        borderColor=PRIMARY,
        borderWidth=1,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_body_style = ParagraphStyle(
        'TableBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
        alignment=0
    )

    table_body_center = ParagraphStyle(
        'TableBodyCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
        alignment=1
    )

    story = []

    # Title Banner
    story.append(Paragraph("Cross-Lingual Prompt-Tuned LLMs for NER in Indian Languages", title_style))
    story.append(Paragraph("<b>Master Technical Specification & End-to-End Evaluation Report</b><br/>Target Languages: Telugu (te) & Kokborok (trp) | Base Model: XLM-RoBERTa (277M)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=10))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Core Objective", h1_style))
    exec_summary = (
        "This project establishes a highly parameter-efficient <b>Multilingual Transfer-Warmed Low-Rank Adaptation (LoRA-r8)</b> framework "
        "for Named Entity Recognition (NER) in low-resource Indian languages. Standard full fine-tuning of 277M parameter models like "
        "<code>xlm-roberta-base</code> suffers from severe data scarcity, high computational overhead, and degradation under agglutinative "
        "morphology. By updating only <b>1,332,487 parameters (0.4780% of total model size)</b> under a 500-shot setting, our proposed method "
        "recovers <b>98.36% of full fine-tuning seqeval F1</b> (0.7668 ± 0.0094 vs. 0.7796) on Telugu and achieves <b>0.7482 F1 (95.42% accuracy)</b> "
        "on Kokborok using a Hugging Face augmented corpus."
    )
    story.append(Paragraph(exec_summary, body_style))
    
    headline_box = (
        "🏆 <b>Key Technical Milestone:</b><br/>"
        "• <b>Telugu Target (500-Shot):</b> 0.7668 ± 0.0094 F1 using 1.33M parameters (0.4780% budget), 5.33 MB adapter checkpoint.<br/>"
        "• <b>Kokborok Target (HF Augmented):</b> 0.7482 F1 (0.7684 Recall / 0.9542 Accuracy) on 4,859 training sentences.<br/>"
        "• <b>Statistical Significance:</b> Paired bootstrap over 10,000 resamples yields p = 0.0001 (95% CI: [0.0035, 0.0221])."
    )
    story.append(Paragraph(headline_box, callout_style))

    # Dataset Specifications
    story.append(Paragraph("2. Dual-Language Dataset Specifications & Split Counts", h1_style))
    story.append(Paragraph("The system was evaluated across exact train, validation, and test splits for both target languages alongside a 600-sentence cross-lingual pre-training warmup corpus:", body_style))

    # Dataset Table
    ds_headers = [Paragraph("Language / Dataset", table_header_style), Paragraph("Split Type", table_header_style), Paragraph("Sentence Count", table_header_style), Paragraph("Entity Labels", table_header_style), Paragraph("Primary Purpose", table_header_style)]
    ds_rows = [
        [Paragraph("<b>Telugu (Naamapadam)</b>", table_body_style), Paragraph("Train (Full Pool)", table_body_center), Paragraph("~400,000", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("Target training candidate pool", table_body_style)],
        [Paragraph("<b>Telugu (Proposed)</b>", table_body_style), Paragraph("<b>Train (500-Shot)</b>", table_body_center), Paragraph("<b>500</b>", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("Proposed 500-shot target setting", table_body_style)],
        [Paragraph("<b>Telugu Validation</b>", table_body_style), Paragraph("Validation (`te_val.json`)", table_body_center), Paragraph("<b>2,700 / 150</b>", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("Ablation & hyperparameter tuning", table_body_style)],
        [Paragraph("<b>Telugu Test Set</b>", table_body_style), Paragraph("<b>Test (`te_test.json`)</b>", table_body_center), Paragraph("<b>847</b>", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("Final test set (~15,000 tokens)", table_body_style)],
        [Paragraph("<b>Kokborok (Native)</b>", table_body_style), Paragraph("Train (`kokborok_train.json`)", table_body_center), Paragraph("11,828", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("Native news training corpus", table_body_style)],
        [Paragraph("<b>Kokborok Test (Native)</b>", table_body_style), Paragraph("Test (`kokborok_test.json`)", table_body_center), Paragraph("3,094 / 69", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("Native news test set evaluation", table_body_style)],
        [Paragraph("<b>Kokborok (HF Augmented)</b>", table_body_style), Paragraph("<b>Train (HF Expansion)</b>", table_body_center), Paragraph("<b>4,859</b>", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("agnivamaiti/kokborok-qa + 273 news", table_body_style)],
        [Paragraph("<b>Kokborok Val / Test (HF)</b>", table_body_style), Paragraph("<b>Val / Test (HF)</b>", table_body_center), Paragraph("<b>500 / 200</b>", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("Hugging Face validation & test sets", table_body_style)],
        [Paragraph("<b>Multilingual Warmup</b>", table_body_style), Paragraph("Stage 1 Pre-Training", table_body_center), Paragraph("<b>600</b>", table_body_center), Paragraph("PER, LOC, ORG", table_body_center), Paragraph("150 En + 150 Hi + 150 Ta + 150 Ml", table_body_style)],
    ]
    
    t_ds = Table([ds_headers] + ds_rows, colWidths=[1.5*inch, 1.3*inch, 1.0*inch, 1.1*inch, 2.3*inch])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 10))

    # Detailed 7-Step Pipeline
    story.append(Paragraph("3. Detailed 7-Step Mechanical Pipeline", h1_style))
    steps = [
        "<b>Step 1: Text Sanitization & Unicode Normalization:</b> Raw inputs are converted via Unicode NFC normalization (<code>unicodedata.normalize('NFC', text)</code>). Invisible artifacts such as Zero-Width Joiners (<code>\\u200d</code>), ZWNJ (<code>\\u200c</code>), Soft Hyphens (<code>\\xad</code>), Byte Order Marks (<code>\\ufeff</code>), and non-standard spaces (<code>\\xa0</code>) are stripped.",
        "<b>Step 2: SentencePiece BPE Subword Deconstruction:</b> Uses <code>xlm-roberta-base</code> (250k vocabulary). Agglutinative words (e.g., Telugu stem <code>దిల్లీ</code> + suffix <code>లో</code> $\\to$ <code>దిల్లీలో</code>) break down into initial stem subword (<code> దిల్లీ</code>) and continuation postposition (<code>లో</code>).",
        "<b>Step 3: Subword BIO Alignment & PyTorch Loss Masking (-100):</b> The initial subword of an entity receives the true BIO label (e.g., <code>B-LOC</code>). Continuation subwords and special tokens (<code>&lt;s&gt;</code>, <code>&lt;/s&gt;</code>) are assigned label <b><code>-100</code></b>. PyTorch <code>nn.CrossEntropyLoss(ignore_index=-100)</code> ignores masked indices, preventing loss distortion.",
        "<b>Step 4: Multilingual Source Transfer Warmup Pre-Training:</b> Pre-trains adapter weights on a 600-sentence balanced corpus (150 English + 150 Hindi + 150 Tamil + 150 Malayalam). Boosts validation F1 from 0.7215 to 0.7584 (+0.0369 gain).",
        "<b>Step 5: Target-Module Expanded LoRA Adapter Computation (r=8):</b> Base transformer weights $W_0 \\in \\mathbb{R}^{768 \\times 768}$ remain <b>FROZEN</b>. Low-rank decomposition matrices $A \\in \\mathbb{R}^{8 \\times 768}$ and $B \\in \\mathbb{R}^{768 \\times 8}$ are injected: $h = W_0 x + \\frac{\\alpha}{r} (B \\cdot A) x$ ($\alpha=16$). Target modules expand to <code>['query', 'key', 'value', 'dense']</code> + classifier head.",
        "<b>Step 6: Softmax Classification & Token Probability Extraction:</b> Sequence representations pass through the classification head generating logits $z_{i,c}$ across 7 classes (<code>O</code>, <code>B-PER</code>, <code>I-PER</code>, <code>B-ORG</code>, <code>I-ORG</code>, <code>B-LOC</code>, <code>I-LOC</code>) evaluated via softmax $P(y_i = c \\mid x) = \\frac{\\exp(z_{i,c})}{\\sum_j \\exp(z_{i,j})}$.",
        "<b>Step 7: Confidence Thresholding (0.50) & Interactive Entity Badging:</b> High-confidence entity predictions ($\ge 50\%$) render solid entity badges (PERSON - Green, LOCATION - Blue, ORGANIZATION - Purple), while lower-confidence predictions trigger warning indicators."
    ]
    for step in steps:
        story.append(Paragraph(step, bullet_style))

    story.append(PageBreak())

    # Visualizations Section
    story.append(Paragraph("4. Visualizations & Graphical Analysis", h1_style))
    story.append(Paragraph("The performance trajectory across shot budgets, parameter efficiency comparison, and validation ablation gains are illustrated below:", body_style))
    story.append(Spacer(1, 4))
    
    story.append(Image(c1, width=6.8*inch, height=3.15*inch))
    story.append(Spacer(1, 6))
    
    img_table_data = [
        [Image(c2, width=3.4*inch, height=1.6*inch), Image(c3, width=3.4*inch, height=1.6*inch)]
    ]
    img_table = Table(img_table_data, colWidths=[3.5*inch, 3.5*inch])
    img_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # Results Section
    story.append(Paragraph("5. Master Comparative Results & Empirical Evaluation", h1_style))
    
    # Telugu Table
    story.append(Paragraph("<b>Target Language 1: Telugu Test Evaluation (N = 847 Sentences)</b>", h2_style))
    te_res_headers = [Paragraph("Model / Architecture", table_header_style), Paragraph("Training Setup", table_header_style), Paragraph("Train Params", table_header_style), Paragraph("Param %", table_header_style), Paragraph("Accuracy", table_header_style), Paragraph("Precision", table_header_style), Paragraph("Recall", table_header_style), Paragraph("seqeval F1", table_header_style)]
    te_res_rows = [
        [Paragraph("IndicNER Baseline", table_body_style), Paragraph("Full Mined (~100k)", table_body_center), Paragraph("278,000,000", table_body_center), Paragraph("100.0%", table_body_center), Paragraph("0.9580", table_body_center), Paragraph("0.8429", table_body_center), Paragraph("0.8477", table_body_center), Paragraph("<b>0.8239</b>", table_body_center)],
        [Paragraph("XLM-R Base (Hindi FT)", table_body_style), Paragraph("Zero-Shot Full FT", table_body_center), Paragraph("277,453,063", table_body_center), Paragraph("100.0%", table_body_center), Paragraph("0.9375", table_body_center), Paragraph("0.7663", table_body_center), Paragraph("0.7932", table_body_center), Paragraph("<b>0.7796</b>", table_body_center)],
        [Paragraph("XLM-R Base (Telugu In-Lang)", table_body_style), Paragraph("In-Language Full FT", table_body_center), Paragraph("277,453,063", table_body_center), Paragraph("100.0%", table_body_center), Paragraph("0.9367", table_body_center), Paragraph("0.7720", table_body_center), Paragraph("0.7682", table_body_center), Paragraph("<b>0.7701</b>", table_body_center)],
        [Paragraph("mBERT (Telugu In-Lang)", table_body_style), Paragraph("In-Language Full FT", table_body_center), Paragraph("177,853,447", table_body_center), Paragraph("100.0%", table_body_center), Paragraph("0.9134", table_body_center), Paragraph("0.6928", table_body_center), Paragraph("0.7233", table_body_center), Paragraph("<b>0.7077</b>", table_body_center)],
        [Paragraph("mBERT (Hindi FT)", table_body_style), Paragraph("Zero-Shot Full FT", table_body_center), Paragraph("177,853,447", table_body_center), Paragraph("100.0%", table_body_center), Paragraph("0.8938", table_body_center), Paragraph("0.6254", table_body_center), Paragraph("0.6742", table_body_center), Paragraph("<b>0.6489</b>", table_body_center)],
        [Paragraph("<b>Proposed LoRA (r=8)</b>", table_body_style), Paragraph("<b>500-Shot Target</b>", table_body_center), Paragraph("<b>1,332,487</b>", table_body_center), Paragraph("<b>0.4780%</b>", table_body_center), Paragraph("0.9290±.003", table_body_center), Paragraph("0.7482±.008", table_body_center), Paragraph("0.7863±.010", table_body_center), Paragraph("<b>0.7668±.009</b>", table_body_center)],
        [Paragraph("Proposed LoRA (r=8)", table_body_style), Paragraph("200-Shot Target", table_body_center), Paragraph("1,332,487", table_body_center), Paragraph("0.4780%", table_body_center), Paragraph("0.9184±.004", table_body_center), Paragraph("0.7125±.011", table_body_center), Paragraph("0.7410±.012", table_body_center), Paragraph("<b>0.7265±.011</b>", table_body_center)],
        [Paragraph("Proposed LoRA (r=8)", table_body_style), Paragraph("100-Shot Target", table_body_center), Paragraph("1,332,487", table_body_center), Paragraph("0.4780%", table_body_center), Paragraph("0.9062±.005", table_body_center), Paragraph("0.6840±.013", table_body_center), Paragraph("0.7180±.014", table_body_center), Paragraph("<b>0.7006±.013</b>", table_body_center)],
        [Paragraph("Proposed LoRA (r=8)", table_body_style), Paragraph("50-Shot Target", table_body_center), Paragraph("1,332,487", table_body_center), Paragraph("0.4780%", table_body_center), Paragraph("0.8920±.006", table_body_center), Paragraph("0.6310±.015", table_body_center), Paragraph("0.6650±.016", table_body_center), Paragraph("<b>0.6475±.015</b>", table_body_center)],
        [Paragraph("Proposed LoRA (r=8)", table_body_style), Paragraph("10-Shot Target", table_body_center), Paragraph("1,332,487", table_body_center), Paragraph("0.4780%", table_body_center), Paragraph("0.8715±.007", table_body_center), Paragraph("0.5240±.018", table_body_center), Paragraph("0.5620±.019", table_body_center), Paragraph("<b>0.5423±.018</b>", table_body_center)],
        [Paragraph("Proposed LoRA (r=8)", table_body_style), Paragraph("0-Shot Target", table_body_center), Paragraph("1,332,487", table_body_center), Paragraph("0.4780%", table_body_center), Paragraph("0.8540±.008", table_body_center), Paragraph("0.4480±.021", table_body_center), Paragraph("0.4850±.022", table_body_center), Paragraph("<b>0.4658±.021</b>", table_body_center)],
    ]

    t_te = Table([te_res_headers] + te_res_rows, colWidths=[1.6*inch, 1.2*inch, 0.95*inch, 0.65*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
    t_te.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor("#E8F5E9")), # highlight proposed
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_te)
    story.append(Spacer(1, 8))

    # Kokborok Table
    story.append(Paragraph("<b>Target Language 2: Kokborok Evaluation Results (Hugging Face Augmented)</b>", h2_style))
    kb_res_headers = [Paragraph("Shot / Setting", table_header_style), Paragraph("Train Sentences", table_header_style), Paragraph("Val / Test Sentences", table_header_style), Paragraph("Trainable Params", table_header_style), Paragraph("Accuracy", table_header_style), Paragraph("Precision", table_header_style), Paragraph("Recall", table_header_style), Paragraph("seqeval F1", table_header_style)]
    kb_res_rows = [
        [Paragraph("0-Shot Baseline", table_body_style), Paragraph("0", table_body_center), Paragraph("50 / 69", table_body_center), Paragraph("1,332,487 (0.478%)", table_body_center), Paragraph("0.2673", table_body_center), Paragraph("0.0039", table_body_center), Paragraph("0.0072", table_body_center), Paragraph("0.0050", table_body_center)],
        [Paragraph("10-Shot Setup", table_body_style), Paragraph("10", table_body_center), Paragraph("50 / 69", table_body_center), Paragraph("1,332,487 (0.478%)", table_body_center), Paragraph("0.8374", table_body_center), Paragraph("0.0000", table_body_center), Paragraph("0.0000", table_body_center), Paragraph("0.0000", table_body_center)],
        [Paragraph("50-Shot Setup", table_body_style), Paragraph("50", table_body_center), Paragraph("50 / 69", table_body_center), Paragraph("1,332,487 (0.478%)", table_body_center), Paragraph("0.8710", table_body_center), Paragraph("0.4120", table_body_center), Paragraph("0.3614", table_body_center), Paragraph("0.3850", table_body_center)],
        [Paragraph("100-Shot Setup", table_body_style), Paragraph("100", table_body_center), Paragraph("50 / 69", table_body_center), Paragraph("1,332,487 (0.478%)", table_body_center), Paragraph("0.8988", table_body_center), Paragraph("0.5333", table_body_center), Paragraph("0.5217", table_body_center), Paragraph("0.5275", table_body_center)],
        [Paragraph("273-Shot Native News", table_body_style), Paragraph("273", table_body_center), Paragraph("50 / 69", table_body_center), Paragraph("1,332,487 (0.478%)", table_body_center), Paragraph("0.9351", table_body_center), Paragraph("0.6442", table_body_center), Paragraph("0.7609", table_body_center), Paragraph("0.6977", table_body_center)],
        [Paragraph("<b>4,859 HF Augmented Peak</b>", table_body_style), Paragraph("<b>4,859</b>", table_body_center), Paragraph("<b>500 / 200</b>", table_body_center), Paragraph("<b>1,332,487 (0.478%)</b>", table_body_center), Paragraph("<b>0.9542</b>", table_body_center), Paragraph("<b>0.7290</b>", table_body_center), Paragraph("<b>0.7684</b>", table_body_center), Paragraph("<b>0.7482</b>", table_body_center)],
    ]

    t_kb = Table([kb_res_headers] + kb_res_rows, colWidths=[1.6*inch, 1.1*inch, 1.2*inch, 1.25*inch, 0.6*inch, 0.65*inch, 0.6*inch, 0.7*inch])
    t_kb.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), ACCENT),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#E0F7FA")), # highlight peak
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_kb)
    story.append(Spacer(1, 10))

    # Statistical Significance & Summary
    story.append(Paragraph("6. Statistical Significance & Conclusion", h1_style))
    stat_text = (
        "• <b>Paired Bootstrap Significance:</b> Evaluated across 10,000 paired sentence resamples between Proposed LoRA ($0.7668$ F1) and XLM-R Full Fine-Tuning ($0.7796$ F1) on N=847 test sentences.<br/>"
        "• <b>Empirical p-value:</b> <b>p = 0.0001</b> (p < 0.05). 95% Confidence Interval: <b>[0.0035, 0.0221]</b> (excludes 0).<br/>"
        "• <b>Conclusion:</b> While the minor 1.28% gap is statistically detectable due to sample size, LoRA-r8 preserves <b>98.36% of full fine-tuning capability</b> while reducing trainable parameter requirements by <b>99.52%</b>, offering an optimal trade-off for low-resource deployment."
    )
    story.append(Paragraph(stat_text, body_style))

    doc.build(story)
    print(f"PDF successfully generated at: {os.path.abspath(pdf_path)}")

if __name__ == "__main__":
    build_pdf()
