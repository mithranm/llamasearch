# Online sources and Gen AI has been used to help with adapting
# the code and fixing minor mistakes

import sqlite3
from pydantic import BaseModel
from typing import Optional, List


# QA - Question and answer
class QARecord(BaseModel):
    question: str
    answer: str
    rating: Optional[int] = 0


conn = sqlite3.connect("qa_data.db")
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS qa_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        rating INTEGER
    )
"""
)


def save_to_db(record: QARecord) -> None:
    conn = sqlite3.connect("qa_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO qa_records (question, answer, rating) VALUES (?, ?, ?)",
        (record.question, record.answer, record.rating),
    )
    conn.commit()
    conn.close()


def load_from_db() -> List[QARecord]:
    conn = sqlite3.connect("qa_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer, rating FROM qa_records")
    records = [
        QARecord(question=row[0], answer=row[1], rating=row[2])
        for row in cursor.fetchall()
    ]
    conn.close()
    return records


def delete_all_records() -> None:
    conn = sqlite3.connect("qa_data.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM qa_records")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='qa_records'")
    conn.commit()
    conn.close()
    print("All records deleted, and ID counter reset to 1.")


def export_to_txt(filename="qa_records.txt") -> None:
    conn = sqlite3.connect("qa_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM qa_records")
    records = cursor.fetchall()
    conn.close()

    rating_map = {1: "Good", -1: "Bad", 0: "N/A"}
    with open(filename, "w", encoding="utf-8") as f:
        for record in records:
            record_id, question, answer, rating = record
            rating_text = rating_map.get(rating, "N/A")
            f.write(f"ID: {record_id}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Answer: {answer}\n")
            f.write(f"Rating: {rating_text}\n")
            f.write("\n")
    print(f"Data exported to {filename}")


if __name__ == "__main__":
    # test
    qa_entry = QARecord(
        question="What is AI?", answer="AI stands for Artificial Intelligence."
    )
    save_to_db(qa_entry)
    print(load_from_db())

    delete_all_records()
    print(load_from_db())

    qa_entry = QARecord(
        question="What is AI?", answer="AI stands for Artificial Intelligence."
    )
    save_to_db(qa_entry)
    print(load_from_db())
    export_to_txt("conversation.txt")
