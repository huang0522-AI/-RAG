from app.chunking import build_chunks, load_documents, split_text

SAMPLE = (
    "新生入学请携带录取通知书与身份证。请在开学前完成线上报到流程。"
    "宿舍分配结果会在迎新系统内统一公布。\n\n"
    "校园卡是校内消费的唯一凭证。遗失后请到一站式服务中心补办。"
    "补办时需携带本人身份证件。临时卡有效期为七天。\n\n"
    "图书馆开放时间为每日八点到二十二点。考试周会延长至二十三点。"
    "进入阅览区请保持安静并出示校园卡。"
)


def test_split_text_respects_size_and_min_length():
    chunks = split_text(SAMPLE, chunk_size=60, overlap=20)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) >= 30
        assert len(chunk) <= 60


def test_split_text_keeps_overlap_between_neighbours():
    chunks = split_text(SAMPLE, chunk_size=60, overlap=20)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-8:] in current


def test_split_text_drops_short_blank_blocks():
    assert split_text("太短。") == []
    assert split_text("   \n\n  ") == []


def test_build_chunks_fields_and_ids(tmp_path):
    doc = tmp_path / "01_新生入学指南.txt"
    doc.write_text(SAMPLE, encoding="utf-8")

    documents = load_documents(tmp_path)
    assert documents == [{"doc": "01_新生入学指南.txt", "text": SAMPLE}]

    chunks = build_chunks(tmp_path, chunk_size=60, overlap=20)
    assert chunks
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))
    first = chunks[0]
    assert first["chunk_id"] == "01_新生入学指南#001"
    assert first["doc"] == "01_新生入学指南.txt"
    assert first["title"] == "新生入学指南"
    assert first["start"] < first["end"]
    for chunk in chunks:
        assert SAMPLE[chunk["start"]:chunk["end"]].strip() == chunk["text"]
