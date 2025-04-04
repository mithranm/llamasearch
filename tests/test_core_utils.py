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


def test_export_multiple_records(tmp_path):
    # Ensure the database is empty before starting.
    delete_all_records()

    # Save two records with distinct ratings.
    record1 = QARecord(question="Q1", answer="A1", rating=1)
    record2 = QARecord(question="Q2", answer="A2", rating=-1)
    save_to_db(record1)
    save_to_db(record2)

    # Export records to a temporary file.
    export_file = tmp_path / "exported_multiple.txt"
    export_to_txt(str(export_file))

    # Verify the file exists and contains expected content.
    assert export_file.exists(), "Export file does not exist."
    content = export_file.read_text(encoding="utf-8")
    assert "Q1" in content, "Record 1 question missing in export."
    assert "A1" in content, "Record 1 answer missing in export."
    assert "Good" in content, "Rating 1 should be mapped to 'Good'."
    assert "Q2" in content, "Record 2 question missing in export."
    assert "A2" in content, "Record 2 answer missing in export."
    assert "Bad" in content, "Rating -1 should be mapped to 'Bad'."


def test_export_empty_db(tmp_path):
    # Ensure the database is empty.
    delete_all_records()

    # Export records to a temporary file.
    export_file = tmp_path / "export_empty.txt"
    export_to_txt(str(export_file))

    # Verify the file exists and that it's empty.
    assert export_file.exists(), "Export file does not exist."
    content = export_file.read_text(encoding="utf-8")
    assert content.strip() == "", "Expected empty export file for an empty database."
