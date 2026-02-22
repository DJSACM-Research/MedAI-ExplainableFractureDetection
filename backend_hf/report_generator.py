import os
import base64
import logging
from io import BytesIO
from datetime import datetime
from typing import Dict, Any
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from textwrap import wrap
import matplotlib.patheffects as pe

logger = logging.getLogger(__name__)

def _b64_to_pil(b64: str) -> Image.Image:
    try:
        return Image.open(BytesIO(base64.b64decode(b64))).convert('RGB')
    except Exception:
        return None

def _make_pdf_report(payload: Dict[str, Any], original_image_bytes: bytes) -> BytesIO:
    """Create a professional multi-page PDF report from the diagnosis payload."""
    buf = BytesIO()

    # ── Design Tokens (aligned with website dark-medical theme) ──────────
    CLR_BG       = '#FFFFFF'
    CLR_HEADER   = '#0f172a'   # slate-900
    CLR_ACCENT   = '#2563eb'   # blue-600 (primary)
    CLR_ACCENT_L = '#dbeafe'   # blue-100
    CLR_TEXT     = '#1e293b'   # slate-800
    CLR_TEXT_SEC = '#64748b'   # slate-500
    CLR_RED      = '#ef4444'   # danger / fracture
    CLR_GREEN    = '#22c55e'   # healthy
    CLR_AMBER    = '#f59e0b'   # amber warning
    CLR_BORDER   = '#e2e8f0'   # slate-200
    CLR_CARD_BG  = '#f8fafc'   # slate-50
    CLR_HEALTHY_BAR = '#22c55e'
    CLR_FRACT_BAR   = '#ef4444'

    FONT_FAMILY  = 'sans-serif'

    # ── Extract all payload data ─────────────────────────────────────────
    pred_data   = payload.get('prediction', {})
    ensemble    = payload.get('ensemble', {})
    explanation = payload.get('explanation', {})
    edu         = payload.get('educational', {}) or {}
    kb          = payload.get('knowledge_base', {}) or {}
    conformal   = payload.get('conformal', {}) or {}
    audit       = payload.get('audit', {}) or {}

    top_class   = pred_data.get('top_class', pred_data.get('class', 'Unknown'))
    top_conf    = pred_data.get('confidence_score', pred_data.get('confidence', 0.0))
    is_fracture = top_class.lower() != 'healthy'

    inference_id = audit.get('inference_id', 'N/A')
    timestamp    = audit.get('timestamp', datetime.utcnow().isoformat())
    try:
        dt_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        date_str = dt_obj.strftime('%B %d, %Y')
        time_str = dt_obj.strftime('%H:%M UTC')
    except Exception:
        date_str = timestamp
        time_str = ''

    # ── Helper Functions ─────────────────────────────────────────────────
    def wrap_text(text, width=80):
        if not text: return ""
        lines = text.split('\n')
        wrapped = []
        for line in lines:
            wrapped.extend(wrap(line, width=width))
        return '\n'.join(wrapped)

    def draw_card(fig, x, y, w, h, fill=CLR_CARD_BG, edge=CLR_BORDER, lw=0.5):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                              facecolor=fill, edgecolor=edge, linewidth=lw,
                              transform=fig.transFigure, clip_on=False)
        fig.patches.append(rect)
        return rect

    try:
        pdf = PdfPages(buf)

        # ═══════════════════════════════════════════════════════════════
        #  PAGE 1: Executive Summary & Clinical Findings
        # ═══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(8.5, 11), facecolor=CLR_BG, dpi=150)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # ── A. Header ────────────────────────────────────────────────
        HEADER_HEIGHT = 0.12
        header_rect = Rectangle((0, 1 - HEADER_HEIGHT), 1, HEADER_HEIGHT, transform=fig.transFigure,
                                facecolor=CLR_HEADER, edgecolor='none', clip_on=False)
        fig.patches.append(header_rect)
        header_stripe = Rectangle((0, 1 - HEADER_HEIGHT - 0.004), 1, 0.004, transform=fig.transFigure,
                                  facecolor=CLR_ACCENT, edgecolor='none', clip_on=False)
        fig.patches.append(header_stripe)

        fig.text(0.05, 0.945, '◈  MedAI', fontsize=24, fontweight='bold',
                 color='white', fontfamily=FONT_FAMILY, va='center')
        fig.text(0.05, 0.915, 'Automated Radiographic Analysis Report',
                 fontsize=12, color='#93c5fd', fontfamily=FONT_FAMILY, va='center')

        fig.text(0.95, 0.955, 'CONFIDENTIAL', fontsize=8, fontweight='bold',
                 color='#f87171', fontfamily=FONT_FAMILY, va='center', ha='right')
        fig.text(0.95, 0.935, f'Date: {date_str}', fontsize=9, color='#cbd5e1',
                 fontfamily=FONT_FAMILY, va='center', ha='right')
        fig.text(0.95, 0.915, f'Time: {time_str}', fontsize=9, color='#cbd5e1',
                 fontfamily=FONT_FAMILY, va='center', ha='right')

        # ── B. Primary Diagnosis Banner ──────────────────────────────
        BANNER_Y = 0.81
        BANNER_H = 0.05
        banner_color = CLR_RED if is_fracture else CLR_GREEN
        banner_bg = '#fef2f2' if is_fracture else '#f0fdf4'
        banner_border = '#fca5a5' if is_fracture else '#86efac'

        draw_card(fig, 0.05, BANNER_Y, 0.90, BANNER_H, fill=banner_bg, edge=banner_border, lw=1)

        status_text = "FRACTURE DETECTED" if is_fracture else "NO FRACTURE DETECTED"
        fig.text(0.07, BANNER_Y + BANNER_H/2, status_text, fontsize=14, fontweight='bold',
                 color=banner_color, fontfamily=FONT_FAMILY, va='center')

        fig.text(0.93, BANNER_Y + BANNER_H/2, f'Confidence: {top_conf*100:.1f}%',
                 fontsize=12, fontweight='bold', color=banner_color,
                 fontfamily=FONT_FAMILY, va='center', ha='right')

        # ── C. Images Section (Original & Grad-CAM) ──────────────────
        IMG_Y = 0.56
        IMG_H = 0.22
        IMG_W = 0.42

        # Original Image
        draw_card(fig, 0.05, IMG_Y, IMG_W, IMG_H)
        fig.text(0.05 + IMG_W/2, IMG_Y + IMG_H + 0.01, 'Original Radiograph',
                 fontsize=9, fontweight='bold', color=CLR_TEXT, fontfamily=FONT_FAMILY, ha='center')

        try:
            orig_img = Image.open(BytesIO(original_image_bytes)).convert('RGB')
            ax1 = fig.add_axes([0.06, IMG_Y + 0.01, IMG_W - 0.02, IMG_H - 0.02])
            ax1.imshow(orig_img)
            ax1.axis('off')
        except Exception:
            fig.text(0.05 + IMG_W/2, IMG_Y + IMG_H/2, '[Image Unavailable]',
                     fontsize=10, color=CLR_TEXT_SEC, ha='center', va='center')

        # Grad-CAM Image
        draw_card(fig, 0.53, IMG_Y, IMG_W, IMG_H)
        fig.text(0.53 + IMG_W/2, IMG_Y + IMG_H + 0.01, 'AI Attention Map (Grad-CAM)',
                 fontsize=9, fontweight='bold', color=CLR_TEXT, fontfamily=FONT_FAMILY, ha='center')

        cam_b64 = explanation.get('heatmap_b64') or explanation.get('primary_heatmap_b64')
        if cam_b64:
            cam_img = _b64_to_pil(cam_b64)
            if cam_img:
                ax2 = fig.add_axes([0.54, IMG_Y + 0.01, IMG_W - 0.02, IMG_H - 0.02])
                ax2.imshow(cam_img)
                ax2.axis('off')
            else:
                fig.text(0.53 + IMG_W/2, IMG_Y + IMG_H/2, '[Heatmap Decode Failed]',
                         fontsize=10, color=CLR_TEXT_SEC, ha='center', va='center')
        else:
            fig.text(0.53 + IMG_W/2, IMG_Y + IMG_H/2, '[Heatmap Unavailable]',
                     fontsize=10, color=CLR_TEXT_SEC, ha='center', va='center')

        # ── D. Clinical Findings (Knowledge Base) ────────────────────
        FIND_Y = 0.38
        FIND_H = 0.14
        draw_card(fig, 0.05, FIND_Y, 0.90, FIND_H)

        fig.text(0.07, FIND_Y + FIND_H - 0.02, 'Clinical Findings',
                 fontsize=11, fontweight='bold', color=CLR_TEXT, fontfamily=FONT_FAMILY)

        diag_name = kb.get('Diagnosis', top_class)
        icd_code  = kb.get('ICD_Code', 'N/A')
        severity  = kb.get('Severity_Rating', 'N/A')
        definition = wrap_text(kb.get('Type_Definition', 'No definition available.'), 110)

        fig.text(0.07, FIND_Y + FIND_H - 0.05, f'Classification: {diag_name}',
                 fontsize=9, fontweight='bold', color=CLR_TEXT, fontfamily=FONT_FAMILY)
        fig.text(0.50, FIND_Y + FIND_H - 0.05, f'ICD-10: {icd_code}',
                 fontsize=9, color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY)
        fig.text(0.75, FIND_Y + FIND_H - 0.05, f'Severity: {severity}',
                 fontsize=9, color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY)

        fig.text(0.07, FIND_Y + FIND_H - 0.08, 'Definition:',
                 fontsize=8, fontweight='bold', color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY)
        fig.text(0.07, FIND_Y + FIND_H - 0.10, definition,
                 fontsize=8, color=CLR_TEXT, fontfamily=FONT_FAMILY, va='top')

        # ── E. Patient Summary (Educational Agent) ───────────────────
        PAT_Y = 0.22
        PAT_H = 0.14
        draw_card(fig, 0.05, PAT_Y, 0.90, PAT_H, fill='#f8fafc', edge=CLR_ACCENT_L, lw=1)

        fig.text(0.07, PAT_Y + PAT_H - 0.02, 'Simplified Explanation',
                 fontsize=11, fontweight='bold', color=CLR_ACCENT, fontfamily=FONT_FAMILY)

        pat_summary = wrap_text(edu.get('patient_summary', 'No summary available.'), 110)
        action_plan = wrap_text(edu.get('next_steps_action_plan', 'Consult physician.'), 110)

        fig.text(0.07, PAT_Y + PAT_H - 0.05, pat_summary,
                 fontsize=8, color=CLR_TEXT, fontfamily=FONT_FAMILY, va='top')

        fig.text(0.07, PAT_Y + 0.06, 'Next Steps / Action Plan:',
                 fontsize=8, fontweight='bold', color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY)
        fig.text(0.07, PAT_Y + 0.04, action_plan,
                 fontsize=8, color=CLR_TEXT, fontfamily=FONT_FAMILY, va='top')

        # ── F. Technical Details (Ensemble & Conformal) ──────────────
        TECH_Y = 0.06
        TECH_H = 0.14
        draw_card(fig, 0.05, TECH_Y, 0.43, TECH_H)
        draw_card(fig, 0.52, TECH_Y, 0.43, TECH_H)

        # Left: Conformal Prediction Set
        fig.text(0.07, TECH_Y + TECH_H - 0.02, 'Conformal Prediction Set',
                 fontsize=9, fontweight='bold', color=CLR_TEXT, fontfamily=FONT_FAMILY)
        fig.text(0.07, TECH_Y + TECH_H - 0.04, 'Statistically guaranteed inclusion set (90% coverage)',
                 fontsize=6, color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY)

        c_set = conformal.get('conformal_set', [])
        if c_set:
            y_offset = TECH_Y + TECH_H - 0.07
            for item in c_set[:4]: # show top 4
                # Handle both dict items and plain string items
                if isinstance(item, dict):
                    c_name = item.get('class', 'Unknown')
                    c_prob = item.get('probability', 0)
                else:
                    c_name = str(item)
                    c_prob = None
                fig.text(0.07, y_offset, f'• {c_name}', fontsize=7, color=CLR_TEXT, fontfamily=FONT_FAMILY)
                if c_prob is not None:
                    fig.text(0.45, y_offset, f'{c_prob*100:.1f}%', fontsize=7, color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY, ha='right')
                y_offset -= 0.015
        else:
            fig.text(0.07, TECH_Y + TECH_H - 0.07, 'Conformal set not generated.',
                     fontsize=7, color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY)

        # Right: Ensemble Models
        fig.text(0.54, TECH_Y + TECH_H - 0.02, 'Ensemble Composition',
                 fontsize=9, fontweight='bold', color=CLR_TEXT, fontfamily=FONT_FAMILY)
        fig.text(0.54, TECH_Y + TECH_H - 0.04, 'Individual model predictions',
                 fontsize=6, color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY)

        indiv_preds = ensemble.get('individual_predictions', {})
        if indiv_preds:
            n_show = min(5, len(indiv_preds))
            row_h = 0.015
            next_y = TECH_Y + TECH_H - 0.06
            # Header row
            fig.text(0.06, next_y - 0.009, 'Model', fontsize=5.5, fontweight='bold',
                     color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY, va='center')
            fig.text(0.48, next_y - 0.009, 'Prediction', fontsize=5.5, fontweight='bold',
                     color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY, va='center', ha='center')
            fig.text(0.88, next_y - 0.009, 'Confidence', fontsize=5.5, fontweight='bold',
                     color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY, va='center', ha='right')
            for i, (mname, mdata) in enumerate(list(indiv_preds.items())[:n_show]):
                row_y = next_y - 0.020 - (i * row_h)
                display_name = mname.replace('_', ' ').replace('best ', '').title()
                m_class = mdata.get('class', '?')
                m_conf  = mdata.get('confidence', 0)
                fig.text(0.06, row_y, display_name, fontsize=6, color=CLR_TEXT,
                         fontfamily=FONT_FAMILY, va='center')
                fig.text(0.48, row_y, m_class, fontsize=6, color=CLR_TEXT,
                         fontfamily=FONT_FAMILY, va='center', ha='center')
                conf_color = CLR_GREEN if m_conf > 0.7 else (CLR_AMBER if m_conf > 0.4 else CLR_RED)
                fig.text(0.88, row_y, f'{m_conf*100:.1f}%', fontsize=6, fontweight='bold',
                         color=conf_color, fontfamily=FONT_FAMILY, va='center', ha='right')

        # ── J. Footer ────────────────────────────────────────────────
        FOOTER_TOP = 0.040
        footer_rect = Rectangle((0, 0), 1, FOOTER_TOP, transform=fig.transFigure,
                                facecolor=CLR_HEADER, edgecolor='none', clip_on=False)
        fig.patches.append(footer_rect)
        footer_stripe = Rectangle((0, FOOTER_TOP), 1, 0.002, transform=fig.transFigure,
                                  facecolor=CLR_ACCENT, edgecolor='none', clip_on=False)
        fig.patches.append(footer_stripe)

        fig.text(0.05, 0.030, '◈ MedAI Research  •  DJSCE-ACM Team',
                 fontsize=6, color='#94a3b8', fontfamily=FONT_FAMILY, va='center')
        fig.text(0.95, 0.030,
                 'AI-assisted analysis — not a substitute for professional medical advice',
                 fontsize=5.5, color='#64748b', fontfamily=FONT_FAMILY, va='center', ha='right')
        fig.text(0.50, 0.012, f'Inference ID: {inference_id}',
                 fontsize=5, color='#475569', fontfamily=FONT_FAMILY, va='center', ha='center')

        pdf.savefig(fig, facecolor=CLR_BG)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════
        #  PAGE 2: AI Explanation (only if Gemini text available)
        # ═══════════════════════════════════════════════════════════════
        gemini_text = kb.get('gemini_explanation')
        if gemini_text:
            fig2 = plt.figure(figsize=(8.5, 11), facecolor=CLR_BG, dpi=150)
            fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Header (same style, page 2)
            h2 = Rectangle((0, 0.925), 1, 0.075, transform=fig2.transFigure,
                            facecolor=CLR_HEADER, edgecolor='none', clip_on=False)
            fig2.patches.append(h2)
            s2 = Rectangle((0, 0.922), 1, 0.004, transform=fig2.transFigure,
                            facecolor=CLR_ACCENT, edgecolor='none', clip_on=False)
            fig2.patches.append(s2)

            fig2.text(0.05, 0.962, '◈  MedAI', fontsize=20, fontweight='bold',
                      color='white', fontfamily=FONT_FAMILY, va='center')
            fig2.text(0.05, 0.940, 'Detailed Clinical Analysis',
                      fontsize=11, color='#93c5fd', fontfamily=FONT_FAMILY, va='center')
            fig2.text(0.95, 0.955, 'Page 2 of 2', fontsize=7, color='#94a3b8',
                      fontfamily=FONT_FAMILY, va='center', ha='right')

            # Gemini explanation content
            fig2.text(0.05, 0.900, 'Detailed Clinical Analysis', fontsize=13, fontweight='bold',
                      color=CLR_TEXT, fontfamily=FONT_FAMILY)
            fig2.text(0.05, 0.886, f'Diagnosis: {top_class}  •  Powered by Gemini',
                      fontsize=8, color=CLR_TEXT_SEC, fontfamily=FONT_FAMILY)

            # Clean and wrap the gemini text
            clean_gemini = gemini_text.replace('**', '').replace('###', '').replace('##', '').replace('#', '')
            wrapped_gemini = wrap_text(clean_gemini, width=105)
            # Limit to fit page
            gemini_lines = wrapped_gemini.split('\n')[:55]

            draw_card(fig2, 0.04, 0.06, 0.92, 0.815, fill='#eef2ff', edge='#c7d2fe', lw=0.8)
            fig2.text(0.06, 0.860, '\n'.join(gemini_lines), fontsize=7, color=CLR_TEXT,
                      fontfamily=FONT_FAMILY, va='top', linespacing=1.5)

            # Page 2 footer
            f2 = Rectangle((0, 0), 1, 0.040, transform=fig2.transFigure,
                            facecolor=CLR_HEADER, edgecolor='none', clip_on=False)
            fig2.patches.append(f2)
            fs2 = Rectangle((0, 0.040), 1, 0.002, transform=fig2.transFigure,
                             facecolor=CLR_ACCENT, edgecolor='none', clip_on=False)
            fig2.patches.append(fs2)
            fig2.text(0.05, 0.025, '◈ MedAI Research  •  DJSCE-ACM Team',
                     fontsize=7, color='#94a3b8', fontfamily=FONT_FAMILY, va='center')
            fig2.text(0.95, 0.025,
                     'AI-assisted analysis — not a substitute for professional medical advice',
                     fontsize=6, color='#64748b', fontfamily=FONT_FAMILY, va='center', ha='right')

            pdf.savefig(fig2, facecolor=CLR_BG)
            plt.close(fig2)

        pdf.close()
        buf.seek(0)
        return buf
    except Exception as e:
        logger.exception('Failed to build PDF report')
        buf.seek(0)
        return buf
