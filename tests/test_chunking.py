from app.chunking import build_chunks, load_documents, split_qa, split_text

QA_SAMPLE = (
    "【问题】国家励志奖学金证书怎么下载？\n"
    "【答复】请通过省教育厅公示名单中的链接，点击“旧版入口”查询个人信息。\n\n"
    "【问题】教资认定普通话要几级？\n"
    "【答复】一般学科需二级乙等，语文学科需二级甲等。"
)


def test_split_qa_keeps_both_markers():
    chunks = split_qa(QA_SAMPLE, max_len=800)
    assert len(chunks) == 2
    for chunk in chunks:
        assert "【问题】" in chunk
        assert "【答复】" in chunk


def test_split_qa_does_not_merge_units():
    chunks = split_qa(QA_SAMPLE, max_len=800)
    assert all(chunk.count("【问题】") == 1 for chunk in chunks)
    assert all(chunk.count("【答复】") == 1 for chunk in chunks)


def test_split_qa_splits_long_unit_and_repeats_question():
    long_answer = "这是相关政策答复内容。" * 80
    text = "【问题】超长的政策问题示例？\n【答复】" + long_answer
    chunks = split_qa(text, max_len=300)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "【问题】" in chunk
        assert "【答复】" in chunk
        assert "超长的政策问题示例" in chunk


def test_build_chunks_qa_path(tmp_path):
    doc = tmp_path / "01_学生资助与奖学.txt"
    doc.write_text(QA_SAMPLE, encoding="utf-8")
    chunks = build_chunks(tmp_path)
    assert len(chunks) == 2
    assert all(
        "【问题】" in chunk["text"] and "【答复】" in chunk["text"] for chunk in chunks
    )
    assert chunks[0]["title"] == "学生资助与奖学"


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
