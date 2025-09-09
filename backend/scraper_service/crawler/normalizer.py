def to_chunks_from_record(record):
    if "id" in record and "text" in record and "metadata" in record:
        return [record]
    text = record.get("text") or str(record)
    return [{"id": record.get("id", "noid"), "text": text, "metadata": record}]
