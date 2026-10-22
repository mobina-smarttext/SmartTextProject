
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "modules" / "plagiarism")
)

from combined_similarity import compare_documents
from sentence_matching import find_similar_sentences


REPORTS_FOLDER = PROJECT_ROOT / "reports"


def generate_report(input_text: str) -> None:
    """ساخت، نمایش و ذخیره گزارش شباهت متون."""

    if not input_text.strip():
        print("خطا: متن ورودی خالی است.")
        return

    print("\nدر حال بررسی متن...")

    # محاسبه نتایج
    document_results = compare_documents(input_text)
    sentence_results = find_similar_sentences(input_text)

    # جمع‌آوری تمام خط‌های گزارش
    lines = []

    lines.append("=" * 55)
    lines.append("گزارش یکپارچه بررسی شباهت متون")
    lines.append("=" * 55)

    lines.append("\nمتن بررسی‌شده:")
    lines.append(input_text)

    lines.append(
        "\nتاریخ و ساعت بررسی: "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    if not document_results:
        lines.append("\nهیچ سندی برای مقایسه وجود ندارد.")

    for document in document_results:
        title = document["title"]

        lines.append("\n" + "-" * 55)
        lines.append(f"نام سند: {title}")
        lines.append("-" * 55)

        lines.append(
            f"شباهت واژگانی: "
            f"{document['lexical_similarity_percent']}٪"
        )

        lines.append(
            f"شباهت معنایی: "
            f"{document['semantic_similarity_percent']}٪"
        )

        lines.append(
            f"امتیاز ترکیبی آزمایشی: "
            f"{document['combined_similarity_percent']}٪"
        )

        matches = [
            item for item in sentence_results
            if item["matched_title"] == title
        ]

        lines.append("\nجمله‌های مشابه:")

        if not matches:
            lines.append(
                "موردی بالاتر از آستانه فعلی پیدا نشد."
            )
        else:
            for number, match in enumerate(matches, start=1):
                lines.append(f"\n  مورد {number}:")

                lines.append(
                    f"  جمله ورودی: "
                    f"{match['input_sentence']}"
                )

                lines.append(
                    f"  جمله ذخیره‌شده: "
                    f"{match['matched_sentence']}"
                )

                lines.append(
                    f"  شباهت معنایی: "
                    f"{match['semantic_similarity_percent']}٪"
                )

    lines.append("\n" + "=" * 55)
    lines.append("پایان گزارش")
    lines.append(
        "توجه: امتیازهای شباهت به‌تنهایی "
        "اثبات سرقت علمی نیستند."
    )
    lines.append("=" * 55)

    report_text = "\n".join(lines)

    # نمایش گزارش در ترمینال
    print("\n" + report_text)

    # ایجاد فولدر گزارش‌ها در صورت نیاز
    REPORTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # نام منحصربه‌فرد برای هر گزارش
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    report_file = (
        REPORTS_FOLDER / f"similarity_report_{timestamp}.txt"
    )

    # ذخیره گزارش با پشتیبانی از زبان فارسی
    report_file.write_text(
        report_text,
        encoding="utf-8-sig"
    )

    print("\nگزارش با موفقیت ذخیره شد.")
    print(f"مسیر فایل: {report_file}")


if __name__ == "__main__":
    print("\nسامانه بررسی شباهت متون")
    print("متن موردنظر خود را وارد کنید.")
    print(
        "برای پایان ورود متن، "
        "در یک خط جداگانه END بنویسید.\n"
    )

    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "END":
            break

        lines.append(line)

    user_text = "\n".join(lines).strip()

    if not user_text:
        print("خطا: متنی وارد نشده است.")
    else:
        generate_report(user_text)
