from llamasearch.ui.utils import (
    QARecord,
    save_to_db,
    load_from_db,
    delete_all_records,
    export_to_txt,
)


# Fixture to clean the database before and after each test.
def clean_db():
    delete_all_records()
    yield
    delete_all_records()


def test_save_and_load():
    # Ensure database is empty
    delete_all_records()
    records = load_from_db()
    assert len(records) == 0, "Database should be empty before saving a record."

    # Save a record
    qa_entry = QARecord(question="Test", answer="Test answer", rating=0)
    save_to_db(qa_entry)

    # Verify that one record exists
    records = load_from_db()
    assert len(records) == 1, f"Expected 1 record, found {len(records)}."

    record = records[0]
    assert record.question == "Test"
    assert record.answer == "Test answer"
    assert record.rating == 0


def test_delete_all_records():
    # Save a record and verify it exists.
    delete_all_records()
    qa_entry = QARecord(question="Test", answer="Test answer", rating=0)
    save_to_db(qa_entry)
    records = load_from_db()
    assert len(records) == 1, f"Expected 1 record, found {len(records)}."

    # Delete all records and verify the database is empty.
    delete_all_records()
    records = load_from_db()
    assert (
        len(records) == 0
    ), f"Expected 0 records after deletion, found {len(records)}."


def test_export_to_txt(tmp_path):
    # Save a record to be exported.
    qa_entry = QARecord(question="Export", answer="Export test", rating=1)
    save_to_db(qa_entry)

    # Export records to a temporary file.
    export_file = tmp_path / "exported.txt"
    export_to_txt(str(export_file))

    # Verify the file exists and contains the expected content.
    assert export_file.exists(), "Export file does not exist."
    content = export_file.read_text(encoding="utf-8")
    assert "Export" in content
    assert "Export test" in content
