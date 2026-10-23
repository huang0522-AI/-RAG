"""切块单元测试：守护 Q&A 感知切块与通用切块两条路径的契约（P2/P9）。

- 通用路径：块长受限、相邻块重叠、丢弃过短/空白块、字段与 chunk_id 正确；
- Q&A 路径：每块同时含【问题】【答复】、两条问答不被合并、超长单元重复问题前缀。
其中 chunk_id/字段/标记类用例属“防回归”，边界（空、过短）属功能验证。
"""

from app.chunking import build_chunks, load_documents, split_qa, split_text

QA_SAMPLE = (
    "【问题】国家励志奖学金证书怎么下载？\n"
    "【答复】请通过省教育厅公示名单中的链接，点击“旧版入口”查询个人信息。\n\n"
    "【问题】教资认定普通话要几级？\n"
    "【答复】一般学科需二级乙等，语文学科需二级甲等。"
)


def test_split_qa_keeps_both_markers():
    """Q&A 切块：每块都同时含【问题】与【答复】。"""
    chunks = split_qa(QA_SAMPLE, max_len=800)
    assert len(chunks) == 2
    for chunk in chunks:
        assert "【问题】" in chunk
        assert "【答复】" in chunk


def test_split_qa_does_not_merge_units():
    """Q&A 切块：一条问答不得被合并进同一块（防回归）。"""
    chunks = split_qa(QA_SAMPLE, max_len=800)
    assert all(chunk.count("【问题】") == 1 for chunk in chunks)
    assert all(chunk.count("【答复】") == 1 for chunk in chunks)


def test_split_qa_splits_long_unit_and_repeats_question():
    """超长问答被切分时，每块都要重复问题前缀。"""
    long_answer = "这是相关政策答复内容。" * 80
    text = "【问题】超长的政策问题示例？\n【答复】" + long_answer
    chunks = split_qa(text, max_len=300)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "【问题】" in chunk
        assert "【答复】" in chunk
        assert "超长的政策问题示例" in chunk


def test_build_chunks_qa_path(tmp_path):
    """build_chunks 对含【问题】的文件走 Q&A 路径，并生成正确 title。"""
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
    """通用切块：每块长度 >= 30 且不超过 chunk_size。"""
    chunks = split_text(SAMPLE, chunk_size=60, overlap=20)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) >= 30
        assert len(chunk) <= 60


def test_split_text_keeps_overlap_between_neighbours():
    """通用切块：相邻块保留重叠（防回归）。"""
    chunks = split_text(SAMPLE, chunk_size=60, overlap=20)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-8:] in current


def test_split_text_drops_short_blank_blocks():
    """过短或纯空白输入返回空列表。"""
    assert split_text("太短。") == []
    assert split_text("   \n\n  ") == []


def test_build_chunks_fields_and_ids(tmp_path):
    """chunk 字段与 chunk_id/start/end 和原文对得上（防回归）。"""
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
