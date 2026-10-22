

from pathlib import Path

from datetime import datetime, timezone, timedelta

import sys

import streamlit as st
# ==========================================# مسیر اصلی پروژه# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(

    0,

    str(PROJECT_ROOT / "modules" / "plagiarism")

)

sys.path.insert(

    0,

    str(PROJECT_ROOT / "modules" / "file_handler")

)
# ==========================================# وارد کردن توابع پروژه# ==========================================
from combined_similarity import compare_documents

from sentence_matching import find_similar_sentences

from text_extractor import extract_text_from_file

from modules.ai_detector.language_detector import (

    detect_language

)

from modules.ai_detector.prediction_service import (

    predict_ai_probability,

    load_calibrated_model

)

from modules.ai_detector.segment_analyzer import (

    analyze_segments

)

from database.history_service import (

    save_ai_history,

    save_plagiarism_history

)

from database.db_manager import (

    get_analysis_history,

    get_analysis_by_id

)

MIN_TEXT_LENGTH = 20

IRAN_TIMEZONE = timezone(

    timedelta(hours=3, minutes=30)

)
# ==========================================# بارگذاری مدل‌ها با کش Streamlit# ==========================================
@st.cache_resource

def get_cached_ai_model(language):

    return load_calibrated_model(language)
# ==========================================# توابع کمکی تاریخچه# ==========================================
def format_history_datetime(utc_datetime):

    """

    Convert SQLite UTC timestamp to Iran time.

    """

    if not utc_datetime:

        return "نامشخص"

    try:

        parsed_datetime = datetime.strptime(

            utc_datetime,

            "%Y-%m-%d %H:%M:%S"

        )

        parsed_datetime = parsed_datetime.replace(

            tzinfo=timezone.utc

        )

        iran_datetime = parsed_datetime.astimezone(

            IRAN_TIMEZONE

        )

        return iran_datetime.strftime(

            "%Y-%m-%d %H:%M:%S"

        )

    except (ValueError, TypeError):

        return str(utc_datetime)

def format_analysis_type(analysis_type):

    labels = {

        "ai_detection": "تشخیص متن هوش مصنوعی",

        "plagiarism": "بررسی شباهت متون",

        "combined": "بررسی ترکیبی"

    }

    return labels.get(

        analysis_type,

        analysis_type

    )

def format_language(language):

    labels = {

        "Persian": "فارسی",

        "English": "انگلیسی",

        "Unknown": "نامشخص"

    }

    return labels.get(

        language,

        language or "نامشخص"

    )

def format_percent(value):

    """

    Format a numeric percentage safely.

    """

    if value is None:

        return "نامشخص"

    try:

        return f"{float(value):.2f}٪"

    except (ValueError, TypeError):

        return "نامشخص"

def display_history_record(record):

    """

    Display stored metadata without original text.

    """

    result = record.get("result", {})

    analysis_type = record.get(

        "analysis_type"

    )

    st.write(

        "**شناسه تحلیل:**",

        record.get("id")

    )

    st.write(

        "**نوع تحلیل:**",

        format_analysis_type(analysis_type)

    )

    st.write(

        "**تاریخ و ساعت ایران:**",

        format_history_datetime(

            record.get("created_at")

        )

    )

    st.write(

        "**زبان متن:**",

        format_language(

            record.get("language")

        )

    )

    st.write(

        "**تعداد کاراکترهای متن:**",

        record.get("text_length")

    )

    st.divider()

    if analysis_type == "ai_detection":

        st.subheader(

            "🤖 نتیجه ذخیره‌شده تشخیص AI"

        )

        prediction = result.get(

            "prediction"

        )

        if prediction == 1:

            st.warning(

                "پیش‌بینی مدل: تولیدشده با هوش مصنوعی"

            )

        elif prediction == 0:

            st.success(

                "پیش‌بینی مدل: نوشته‌شده توسط انسان"

            )

        else:

            st.info(

                "دسته پیش‌بینی‌شده ثبت نشده است."

            )

        col1, col2 = st.columns(2)

        ai_probability = result.get(

            "ai_probability"

        )

        human_probability = result.get(

            "human_probability"

        )

        col1.metric(

            "احتمال تخمینی AI",

            (

                format_percent(

                    ai_probability * 100

                )

                if ai_probability is not None

                else "نامشخص"

            )

        )

        col2.metric(

            "احتمال تخمینی انسانی",

            (

                format_percent(

                    human_probability * 100

                )

                if human_probability is not None

                else "نامشخص"

            )

        )

        mixed_language = result.get(

            "mixed_language"

        )

        st.write(

            "**متن ترکیبی فارسی و انگلیسی:**",

            "بله" if mixed_language else "خیر"

        )

        segments = result.get(

            "segments"

        )

        if isinstance(segments, dict):

            st.divider()

            st.subheader(

                "📄 خلاصه بررسی پاراگراف‌ها"

            )

            col_a, col_b, col_c = st.columns(3)

            col_a.metric(

                "کل بخش‌ها",

                segments.get("total", 0)

            )

            col_b.metric(

                "بخش‌های بررسی‌شده",

                segments.get("analyzed", 0)

            )

            col_c.metric(

                "پیش‌بینی‌شده به‌عنوان AI",

                segments.get("predicted_ai", 0)

            )

            st.write(

                "**پیش‌بینی‌شده به‌عنوان انسانی:**",

                segments.get(

                    "predicted_human",

                    0

                )

            )

            status_counts = segments.get(

                "status_counts",

                {}

            )

            if status_counts:

                st.write(

                    "**وضعیت بخش‌ها:**"

                )

                status_labels = {

                    "analyzed": "بررسی‌شده",

                    "too_short": "خیلی کوتاه",

                    "language_mismatch": (

                        "تفاوت یا ترکیب زبان"

                    )

                }

                for status, count in status_counts.items():

                    st.write(

                        f"- {status_labels.get(status, status)}: "

                        f"{count}"

                    )

    elif analysis_type == "plagiarism":

        st.subheader(

            "📚 نتیجه ذخیره‌شده بررسی شباهت"

        )

        col1, col2, col3 = st.columns(3)

        col1.metric(

            "اسناد مقایسه‌شده",

            result.get(

                "compared_documents",

                0

            )

        )

        col2.metric(

            "جمله‌های مشابه",

            result.get(

                "similar_sentence_count",

                0

            )

        )

        col3.metric(

            "بیشترین امتیاز ترکیبی",

            format_percent(

                result.get(

                    "max_combined_similarity_percent"

                )

            )

        )

        document_scores = result.get(

            "document_scores",

            []

        )

        if document_scores:

            st.divider()

            st.write(

                "**امتیازهای اسناد مقایسه‌شده:**"

            )

            for index, document in enumerate(

                document_scores,

                start=1

            ):

                with st.expander(

                    f"سند شماره {index}"

                ):

                    st.write(

                        "شباهت واژگانی:",

                        format_percent(

                            document.get(

                                "lexical_similarity_percent"

                            )

                        )

                    )

                    st.write(

                        "شباهت معنایی:",

                        format_percent(

                            document.get(

                                "semantic_similarity_percent"

                            )

                        )

                    )

                    st.write(

                        "امتیاز ترکیبی آزمایشی:",

                        format_percent(

                            document.get(

                                "combined_similarity_percent"

                            )

                        )

                    )

    else:

        st.info(

            "برای این نوع تحلیل، نمایش اختصاصی "

            "هنوز تعریف نشده است."

        )

    st.caption(

        "این اطلاعات خلاصه ذخیره‌شده تحلیل هستند. "

        "متن اصلی و گزارش کامل در تاریخچه نگهداری نشده‌اند."

    )
# ==========================================# ساخت گزارش شباهت متون# ==========================================
def create_similarity_report(

    input_text,

    document_results,

    sentence_results

):

    lines = [

        "گزارش بررسی شباهت متون",

        "=" * 50,

        "",

        "متن بررسی‌شده:",

        input_text,

        "",

        "تاریخ بررسی: "

        + datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        ),

    ]

    for document in document_results:

        title = document["title"]

        lines.extend([

            "",

            "-" * 50,

            f"نام سند: {title}",

            f"شباهت واژگانی: "

            f"{document['lexical_similarity_percent']}٪",

            f"شباهت معنایی: "

            f"{document['semantic_similarity_percent']}٪",

            f"امتیاز ترکیبی آزمایشی: "

            f"{document['combined_similarity_percent']}٪",

            "",

            "جمله‌های مشابه:"

        ])

        matches = [

            item for item in sentence_results

            if item["matched_title"] == title

        ]

        if not matches:

            lines.append(

                "موردی بالاتر از آستانه فعلی پیدا نشد."

            )

        else:

            for match in matches:

                lines.extend([

                    "",

                    f"جمله ورودی: {match['input_sentence']}",

                    f"جمله ذخیره‌شده: {match['matched_sentence']}",

                    f"شباهت معنایی: "

                    f"{match['semantic_similarity_percent']}٪"

                ])

    lines.extend([

        "",

        "=" * 50,

        "توجه: شباهت متنی به‌تنهایی "

        "اثبات سرقت علمی نیست."

    ])

    return "\n".join(lines)
# ==========================================# ساخت گزارش تشخیص هوش مصنوعی# ==========================================
def create_ai_report(

    input_text,

    language_label,

    language,

    mixed_language,

    prediction,

    ai_probability,

    human_probability,

    segment_result=None

):

    result_label = (

        "تولیدشده با هوش مصنوعی"

        if prediction == 1

        else "نوشته‌شده توسط انسان"

    )

    report_date = datetime.now().strftime(

        "%Y-%m-%d %H:%M:%S"

    )

    lines = [

        "گزارش تشخیص متن تولیدشده با هوش مصنوعی",

        "=" * 50,

        "",

        "متن بررسی‌شده:",

        input_text,

        "",

        f"تاریخ بررسی: {report_date}",

        "",

        f"زبان غالب تشخیص‌داده‌شده: {language_label}",

        "",

        f"مدل استفاده‌شده: Calibrated SVM - {language}",

        "",

        f"نتیجه پیش‌بینی مدل: {result_label}",

        "",

        f"احتمال تخمینی AI: {ai_probability * 100:.2f}٪",

        f"احتمال تخمینی انسانی: {human_probability * 100:.2f}٪",

        "",

        "متن ترکیبی فارسی و انگلیسی: "

        + ("بله" if mixed_language else "خیر"),

    ]

    if abs(ai_probability - 0.5) < 0.10:

        lines.extend([

            "",

            "هشدار: احتمال‌های دو کلاس نزدیک هستند.",

            "نتیجه در محدوده مرزی مدل قرار دارد."

        ])

    lines.extend([

        "",

        "=" * 50,

        "نتایج بررسی جداگانه بخش‌های متن",

        "=" * 50,

        "",

        "توجه: امتیازهای پاراگراف‌ها آزمایشی هستند "

        "و هنوز اعتبارسنجی مستقل نشده‌اند."

    ])

    if segment_result is None:

        lines.extend([

            "",

            "نتیجه بررسی جداگانه بخش‌ها در دسترس نیست."

        ])

    else:

        lines.extend([

            "",

            "تعداد بخش‌های شناسایی‌شده: "

            f"{segment_result['total_segments']}"

        ])

        for segment in segment_result["segments"]:

            segment_number = segment["segment_number"]

            segment_text = segment["text"]

            word_count = segment["word_count"]

            status = segment["status"]

            lines.extend([

                "",

                "-" * 50,

                f"بخش شماره {segment_number}",

                "-" * 50,

                "",

                "متن بخش:",

                segment_text,

                "",

                f"تعداد کلمات: {word_count}"

            ])

            if status == "too_short":

                lines.append(

                    "وضعیت: متن این بخش برای بررسی "

                    "مستقل خیلی کوتاه است."

                )

            elif status == "language_mismatch":

                lines.append(

                    "وضعیت: زبان بخش متفاوت یا ترکیبی "

                    "است و این بخش بررسی نشده است."

                )

            elif status == "analyzed":

                segment_ai = segment["ai_probability"]

                segment_human = segment["human_probability"]

                segment_prediction = segment["prediction"]

                segment_label = (

                    "تولیدشده با هوش مصنوعی"

                    if segment_prediction == 1

                    else "نوشته‌شده توسط انسان"

                )

                lines.extend([

                    "وضعیت: بررسی‌شده",

                    f"امتیاز آزمایشی AI: "

                    f"{segment_ai * 100:.2f}٪",

                    f"امتیاز آزمایشی انسانی: "

                    f"{segment_human * 100:.2f}٪",

                    f"دسته پیش‌بینی‌شده: {segment_label}"

                ])

                if segment_ai >= 0.80:

                    lines.append(

                        "علامت‌گذاری آزمایشی: "

                        "امتیاز AI بالا؛ "

                        "نیازمند بررسی بیشتر."

                    )

                elif segment_ai >= 0.50:

                    lines.append(

                        "علامت‌گذاری آزمایشی: "

                        "امتیاز AI بالاتر از ۵۰٪؛ "

                        "نتیجه قطعی نیست."

                    )

                else:

                    lines.append(

                        "علامت‌گذاری آزمایشی: "

                        "امتیاز AI کمتر از ۵۰٪."

                    )

            else:

                lines.append(

                    f"وضعیت: {status}"

                )

    lines.extend([

        "",

        "=" * 50,

        "توجه و محدودیت‌های سامانه:",

        "",

        "۱. احتمال‌های نمایش‌داده‌شده برآورد مدل هستند.",

        "۲. این نتیجه اثبات قطعی درباره نویسنده متن نیست.",

        "۳. امتیازهای بخش‌های متن هنوز به‌صورت مستقل "

        "اعتبارسنجی نشده‌اند.",

        "۴. آستانه ۸۰٪ برای علامت‌گذاری بخش‌ها "

        "صرفاً آزمایشی است.",

        "۵. دقت مدل روی مجموعه آزمون به معنی "

        "دقت تضمین‌شده روی تمام متن‌های جدید نیست.",

        "۶. امتیاز کل متن لزوماً برابر میانگین "

        "امتیاز پاراگراف‌ها نیست."

    ])

    return "\n".join(lines)
# ==========================================# تنظیمات صفحه Streamlit# ==========================================
st.set_page_config(

    page_title="سامانه هوشمند بررسی متون",

    page_icon="📚",

    layout="wide"

)
# ==========================================# تنظیم راست‌چین بودن صفحه# ==========================================
st.markdown("""
<style>
/* Light and dark mode compatible, RTL-friendly interface */
.stApp { direction: rtl; }
.block-container { max-width: 1160px; padding-top: 1.7rem; padding-bottom: 3rem; }
h1, h2, h3, h4, h5, h6, p, label { direction: rtl; text-align: right; }
textarea { direction: auto !important; line-height: 1.9 !important; }
[data-testid="stAlert"], [data-testid="stMetric"], [data-testid="stFileUploader"] {
    direction: rtl; text-align: right;
}
[data-testid="stMetric"] {
    padding: 1rem 1.15rem; border-radius: 15px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    background: rgba(128, 128, 128, 0.055);
}
[data-testid="stMetricValue"] { font-weight: 750; }
.stTabs [data-baseweb="tab-list"] {
    gap: .65rem; border-bottom: 1px solid rgba(128,128,128,.22);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px 12px 0 0; padding: .7rem 1.25rem;
    font-weight: 650;
}
.stButton button[kind="primary"] { border-radius: 12px; min-height: 2.8rem; font-weight: 700; }
[data-testid="stExpander"] { border-radius: 13px; overflow: hidden; }
.smarttext-hero {
    border: 1px solid rgba(128,128,128,.24); border-radius: 20px;
    padding: 1.4rem 1.65rem; margin-bottom: 1.1rem;
    background: linear-gradient(115deg, rgba(49,110,198,.10), rgba(15,168,141,.055));
}
.smarttext-hero h1 { font-size: clamp(1.45rem, 3vw, 2.15rem); margin: 0 0 .5rem; }
.smarttext-hero p { opacity: .82; margin: 0; line-height: 1.9; }
.smarttext-eyebrow { font-size: .82rem; font-weight: 750; letter-spacing: .02em; margin-bottom: .5rem; }
.smarttext-footnote { opacity: .7; font-size: .84rem; text-align: center; padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================# عنوان اصلی سامانه# ==========================================
st.markdown("""
<div class="smarttext-hero">
  <div class="smarttext-eyebrow">🎓 پروژه کارشناسی مهندسی کامپیوتر</div>
  <h1>📚 سامانه هوشمند بررسی متون</h1>
  <p>بررسی شباهت واژگانی و معنایی متون، تشخیص آزمایشی متن تولیدشده با هوش مصنوعی،
  تحلیل پاراگراف‌ها و نگهداری خلاصه نتایج در تاریخچه محلی</p>
</div>
""", unsafe_allow_html=True)

# ==========================================# ساخت تب‌های اصلی# ==========================================
analysis_tab, history_tab = st.tabs([

    "🔍 بررسی متن",

    "🗂️ تاریخچه تحلیل‌ها"

])
# ==========================================# تب اول: بررسی متن# ==========================================
with analysis_tab:

    st.caption("مرحله ۱: متن یا فایل را وارد کنید  ←  مرحله ۲: نوع بررسی را انتخاب کنید  ←  مرحله ۳: نتیجه و گزارش را دریافت کنید")

    st.subheader(

        "متن موردنظر خود را وارد کنید"

    )
# --------------------------------------# دریافت متن از کاربر# --------------------------------------
    user_text = st.text_area(

        "متن ورودی",

        height=200,

        placeholder=(

            "متن فارسی یا انگلیسی خود را وارد کنید..."

        )

    )
# --------------------------------------# دریافت فایل از کاربر# --------------------------------------
    uploaded_file = st.file_uploader(

        "📂 یا فایل خود را بارگذاری کنید",

        type=["txt", "pdf", "docx"],

        help=(

            "فایل‌های TXT، PDF متنی و DOCX "

            "پشتیبانی می‌شوند."

        )

    )

    st.caption(

        "اگر فایل بارگذاری کنید، متن فایل "

        "به‌جای متن تایپ‌شده بررسی می‌شود."

    )
    with st.expander("ℹ️ راهنمای ورودی و نحوه تفسیر نتایج", expanded=False):
        st.markdown("""
- متن فارسی یا انگلیسی را وارد کنید یا فایل **TXT، PDF متنی یا DOCX** بارگذاری کنید.
- اگر فایل بارگذاری شود، محتوای آن جایگزین متن تایپ‌شده می‌شود.
- درصد تشخیص AI یک **برآورد مدل** است، نه اثبات قطعی نویسندگی.
- شباهت متنی به‌تنهایی اثبات سرقت علمی نیست.
- فقط خلاصه عددی نتایج در تاریخچه محلی ذخیره می‌شود.
""")

    # --------------------------------------# انتخاب نوع بررسی# --------------------------------------
    analysis_type = st.radio(

        "نوع بررسی را انتخاب کنید:",

        [

            "بررسی شباهت متون",

            "تشخیص متن تولیدشده با هوش مصنوعی"

        ]

    )

    st.caption(

        "پس از انجام موفق بررسی، خلاصه عددی نتیجه "

        "و زمان آن در پایگاه داده محلی ذخیره می‌شود. "

        "متن کامل، نام فایل و گزارش دانلودی "

        "در تاریخچه ذخیره نمی‌شوند."

    )
# ======================================# شروع بررسی# ======================================
    if st.button(

        "🔍 شروع بررسی",

        type="primary"

    ):

        clean_text = user_text.strip()

        file_error = None
# ----------------------------------# استخراج متن فایل# ----------------------------------
        if uploaded_file is not None:

            try:

                clean_text = extract_text_from_file(

                    uploaded_file

                ).strip()

            except ValueError as error:

                file_error = str(error)
# ----------------------------------# اعتبارسنجی ورودی# ----------------------------------
        if file_error:

            st.error(file_error)

        elif not clean_text:

            st.warning(

                "لطفاً متن وارد کنید یا "

                "یک فایل دارای متن بارگذاری کنید."

            )

        elif len(clean_text) < MIN_TEXT_LENGTH:

            st.warning(

                "متن واردشده خیلی کوتاه است. "

                f"حداقل {MIN_TEXT_LENGTH} کاراکتر لازم است."

            )
# ==================================# بررسی شباهت متون# ==================================
        elif analysis_type == "بررسی شباهت متون":

            with st.spinner(

                "در حال بررسی شباهت متون..."

            ):

                document_results = compare_documents(

                    clean_text

                )

                sentence_results = find_similar_sentences(

                    clean_text

                )

            st.success(

                "بررسی شباهت متون انجام شد."

            )
# ------------------------------# ذخیره خلاصه در SQLite# ------------------------------
            try:

                detected_language, _ = detect_language(

                    clean_text

                )

                history_id = save_plagiarism_history(

                    language=detected_language,

                    text_length=len(clean_text),

                    document_results=document_results,

                    sentence_results=sentence_results,

                )

                st.caption(

                    "نتیجه این بررسی در تاریخچه ذخیره شد. "

                    f"شناسه: {history_id}"

                )

            except Exception as error:

                st.warning(

                    "بررسی شباهت انجام شد، اما ذخیره "

                    "تاریخچه با مشکل مواجه شد."

                )

                print(

                    "Plagiarism history save error:",

                    repr(error)

                )

            if not document_results:

                st.info(

                    "هیچ سندی برای مقایسه وجود ندارد."

                )

            for document in document_results:

                title = document["title"]

                st.subheader(f"📄 {title}")

                col1, col2, col3 = st.columns(3)

                col1.metric(

                    "شباهت واژگانی",

                    f"{document['lexical_similarity_percent']}٪"

                )

                col2.metric(

                    "شباهت معنایی",

                    f"{document['semantic_similarity_percent']}٪"

                )

                col3.metric(

                    "امتیاز ترکیبی آزمایشی",

                    f"{document['combined_similarity_percent']}٪"

                )

                matches = [

                    item for item in sentence_results

                    if item["matched_title"] == title

                ]

                if matches:

                    with st.expander(

                        "🔎 مشاهده جمله‌های مشابه"

                    ):

                        for match in matches:

                            st.write(

                                "**جمله ورودی:**",

                                match["input_sentence"]

                            )

                            st.write(

                                "**جمله ذخیره‌شده:**",

                                match["matched_sentence"]

                            )

                            st.write(

                                "**شباهت معنایی:**",

                                f"{match['semantic_similarity_percent']}٪"

                            )

                            st.divider()

                else:

                    st.caption(

                        "جمله‌ای بالاتر از آستانه فعلی "

                        "پیدا نشد."

                    )
# ------------------------------# دانلود گزارش شباهت# ------------------------------
            report_text = create_similarity_report(

                clean_text,

                document_results,

                sentence_results

            )

            st.download_button(

                label="📥 دانلود گزارش شباهت متون",

                data=report_text.encode("utf-8-sig"),

                file_name="similarity_report.txt",

                mime="text/plain"

            )

            st.warning(

                "توجه: امتیازهای شباهت به‌تنهایی "

                "اثبات سرقت علمی نیستند."

            )
# ==================================# تشخیص متن AI# ==================================
        else:

            language, mixed_language = detect_language(

                clean_text

            )

            if language == "Unknown":

                st.warning(

                    "زبان متن قابل تشخیص نیست. "

                    "لطفاً متن فارسی یا انگلیسی وارد کنید."

                )

            else:

                language_label = (

                    "فارسی"

                    if language == "Persian"

                    else "انگلیسی"

                )

                st.info(

                    "🌐 زبان غالب تشخیص‌داده‌شده: "

                    f"{language_label}"

                )

                if mixed_language:

                    st.warning(

                        "متن شامل مقدار قابل‌توجهی از هر دو "

                        "زبان فارسی و انگلیسی است. "

                        "مدل بر اساس زبان غالب انتخاب می‌شود "

                        "و نتیجه برای متن‌های ترکیبی "

                        "هنوز اعتبارسنجی نشده است."

                    )
# --------------------------# پیش‌بینی کلی مدل# --------------------------
                try:

                    with st.spinner(

                        "در حال بررسی متن با مدل "

                        "کالیبره‌شده SVM..."

                    ):

                        model = get_cached_ai_model(

                            language

                        )

                        result = predict_ai_probability(

                            clean_text,

                            model=model

                        )

                        prediction = result[

                            "prediction"

                        ]

                        ai_probability = result[

                            "ai_probability"

                        ]

                        human_probability = result[

                            "human_probability"

                        ]

                except (

                    FileNotFoundError,

                    OSError,

                    ValueError

                ) as error:

                    st.error(

                        "اجرای مدل امکان‌پذیر نیست: "

                        + str(error)

                    )

                else:

                    st.subheader(

                        "🤖 نتیجه تشخیص متن"

                    )

                    col1, col2 = st.columns(2)

                    col1.metric(

                        "🤖 احتمال تخمینی AI",

                        f"{ai_probability * 100:.2f}٪"

                    )

                    col2.metric(

                        "👤 احتمال تخمینی انسانی",

                        f"{human_probability * 100:.2f}٪"

                    )

                    st.progress(

                        ai_probability,

                        text="میزان احتمال تخمینی AI"

                    )

                    if abs(ai_probability - 0.5) < 0.10:

                        st.info(

                            "احتمال‌های دو کلاس به هم "

                            "نزدیک هستند. این پیش‌بینی "

                            "در محدوده مرزی مدل قرار دارد "

                            "و باید با احتیاط تفسیر شود."

                        )

                    if prediction == 1:

                        st.warning(

                            "مدل این متن را در دسته "

                            "«تولیدشده با هوش مصنوعی» "

                            "قرار داده است."

                        )

                    else:

                        st.success(

                            "مدل این متن را در دسته "

                            "«نوشته‌شده توسط انسان» "

                            "قرار داده است."

                        )

                    st.info(

                        "این نتیجه صرفاً پیش‌بینی مدل است "

                        "و اثبات قطعی درباره نویسنده متن نیست."

                    )
# ======================# بررسی پاراگراف‌ها# ======================
                    st.divider()

                    st.subheader(

                        "🔎 بررسی جداگانه بخش‌های متن"

                    )

                    st.caption(

                        "این قابلیت آزمایشی است. امتیازهای "

                        "پاراگراف‌ها هنوز به‌صورت مستقل "

                        "اعتبارسنجی نشده‌اند و نباید "

                        "اثبات تولید متن توسط AI تلقی شوند."

                    )

                    segment_result = None

                    try:

                        with st.spinner(

                            "در حال بررسی پاراگراف‌ها..."

                        ):

                            segment_result = analyze_segments(

                                clean_text

                            )

                    except (

                        FileNotFoundError,

                        OSError,

                        ValueError

                    ) as error:

                        st.error(

                            "بررسی بخش‌های متن امکان‌پذیر نیست: "

                            + str(error)

                        )

                    else:

                        st.write(

                            "تعداد بخش‌های شناسایی‌شده:",

                            segment_result["total_segments"]

                        )

                        for segment in segment_result["segments"]:

                            segment_number = segment[

                                "segment_number"

                            ]

                            with st.expander(

                                f"📄 بخش {segment_number}",

                                expanded=True

                            ):

                                st.write(

                                    segment["text"]

                                )

                                st.caption(

                                    "تعداد کلمات: "

                                    f"{segment['word_count']}"

                                )

                                if segment["status"] == "too_short":

                                    st.info(

                                        "این بخش برای تحلیل "

                                        "مستقل خیلی کوتاه است."

                                    )

                                elif segment["status"] == "language_mismatch":

                                    st.warning(

                                        "زبان این بخش با زبان "

                                        "غالب متن متفاوت است "

                                        "یا ترکیبی است؛ "

                                        "بنابراین بررسی نشده."

                                    )

                                elif segment["status"] == "analyzed":

                                    segment_ai = segment[

                                        "ai_probability"

                                    ]

                                    segment_human = segment[

                                        "human_probability"

                                    ]

                                    col_a, col_b = st.columns(2)

                                    col_a.metric(

                                        "🤖 امتیاز آزمایشی AI",

                                        f"{segment_ai * 100:.2f}٪"

                                    )

                                    col_b.metric(

                                        "👤 امتیاز آزمایشی انسانی",

                                        f"{segment_human * 100:.2f}٪"

                                    )

                                    st.progress(

                                        segment_ai

                                    )

                                    if segment_ai >= 0.80:

                                        st.warning(

                                            "این بخش امتیاز AI "

                                            "بالایی دریافت کرده "

                                            "و برای بررسی بیشتر "

                                            "علامت‌گذاری شده است."

                                        )

                                    elif segment_ai >= 0.50:

                                        st.info(

                                            "این بخش امتیاز AI "

                                            "بالاتر از ۵۰٪ دارد، "

                                            "اما نتیجه قطعی نیست."

                                        )

                                    else:

                                        st.caption(

                                            "امتیاز AI این بخش "

                                            "کمتر از ۵۰٪ است."

                                        )
# ----------------------# ساخت گزارش AI# ----------------------
                    ai_report = create_ai_report(

                        input_text=clean_text,

                        language_label=language_label,

                        language=language,

                        mixed_language=mixed_language,

                        prediction=prediction,

                        ai_probability=ai_probability,

                        human_probability=human_probability,

                        segment_result=segment_result

                    )
# ----------------------# ذخیره خلاصه در SQLite# ----------------------
                    try:

                        history_id = save_ai_history(

                            language=language,

                            text_length=len(clean_text),

                            prediction=prediction,

                            ai_probability=ai_probability,

                            human_probability=human_probability,

                            mixed_language=mixed_language,

                            segment_result=segment_result,

                        )

                        st.caption(

                            "نتیجه این بررسی در تاریخچه ذخیره شد. "

                            f"شناسه: {history_id}"

                        )

                    except Exception as error:

                        st.warning(

                            "تشخیص متن انجام شد، اما ذخیره "

                            "تاریخچه با مشکل مواجه شد."

                        )

                        print(

                            "AI history save error:",

                            repr(error)

                        )
# ----------------------# دانلود گزارش AI# ----------------------
                    st.download_button(

                        label=(

                            "📥 دانلود گزارش تشخیص "

                            "هوش مصنوعی"

                        ),

                        data=ai_report.encode("utf-8-sig"),

                        file_name="ai_detection_report.txt",

                        mime="text/plain"

                    )
# ==========================================# تب دوم: تاریخچه تحلیل‌ها# ==========================================
with history_tab:

    st.subheader("🗂️ تاریخچه تحلیل‌های ذخیره‌شده")
    st.caption("برای دیدن جزئیات، یکی از تحلیل‌های ثبت‌شده را از فهرست انتخاب کنید.")

    st.caption(

        "در این بخش خلاصه تحلیل‌های ذخیره‌شده "

        "در پایگاه داده محلی نمایش داده می‌شود. "

        "متن اصلی و گزارش کامل ذخیره نشده‌اند."

    )

    try:

        history = get_analysis_history(

            limit=50

        )

    except Exception as error:

        st.error(

            "خواندن تاریخچه تحلیل‌ها امکان‌پذیر نیست."

        )

        print(

            "History retrieval error:",

            repr(error)

        )

    else:

        if not history:

            st.info(

                "هنوز هیچ تحلیلی در تاریخچه ثبت نشده است."

            )

        else:

            st.write(

                "تعداد رکوردهای نمایش‌داده‌شده:",

                len(history)

            )

            st.caption(

                "حداکثر ۵۰ تحلیل اخیر نمایش داده می‌شود."

            )

            history_options = [

                row["id"]

                for row in history

            ]

            history_lookup = {

                row["id"]: row

                for row in history

            }

            selected_id = st.selectbox(

                "یک تحلیل را برای مشاهده جزئیات انتخاب کنید:",

                options=history_options,

                format_func=lambda record_id: (

                    f"#{record_id} | "

                    f"{format_analysis_type(history_lookup[record_id]['analysis_type'])}"

                    f" | "

                    f"{format_history_datetime(history_lookup[record_id]['created_at'])}"

                )

            )

            if selected_id is not None:

                try:

                    selected_record = get_analysis_by_id(

                        selected_id

                    )

                except Exception as error:

                    st.error(

                        "خواندن جزئیات این تحلیل "

                        "امکان‌پذیر نیست."

                    )

                    print(

                        "History detail error:",

                        repr(error)

                    )

                else:

                    if selected_record is None:

                        st.warning(

                            "رکورد انتخاب‌شده پیدا نشد."

                        )

                    else:

                        st.divider()

                        display_history_record(

                            selected_record

                        )

    st.divider()

    st.caption(

        "توجه: تاریخچه فعلی مخصوص اجرای محلی پروژه است. "

        "برای استفاده چندکاربره، احراز هویت و "

        "تفکیک تاریخچه کاربران ضروری خواهد بود."

    )


st.markdown('<div class="smarttext-footnote">SmartText · ابزار کمک‌تحلیلی پژوهشی؛ تصمیم نهایی نیازمند بررسی انسانی است.</div>', unsafe_allow_html=True)
