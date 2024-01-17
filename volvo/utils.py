def remove_sidebars(elements):
    from unstructured.documents.elements import ElementType
    from collections import defaultdict
    import re

    if not elements:
        return elements
    points_groups = defaultdict(list)
    min_x = 99999999
    max_x = 0
    e2index = {e.id: i for i, e in enumerate(elements)}
    for e in elements:
        x_l = int(e.metadata.coordinates.points[0][0])
        x_r = int(e.metadata.coordinates.points[2][0])
        points_groups[(x_l, x_r)].append(e)
        min_x = min(min_x, x_l)
        max_x = max(max_x, x_r)
    sidebars_elements = set()
    for (x_l, x_r), es in points_groups.items():
        first_id = e2index[es[0].id]
        last_id = e2index[es[-1].id]
        on_left = first_id == 0 and x_l == min_x
        on_right = (last_id == len(elements) - 2) and x_r == max_x
        loc_match = [on_left, on_right]
        total_text = "".join(map(str, es))
        condiction = [
            any(loc_match),
            len(es) >= 3,
            re.findall("^[A-Z\s\d,]+$", total_text),
        ]
        if not all(condiction):
            continue
        sidebars_elements.update(map(lambda x: x.id, es))
        if on_left:
            check_page_num_e = elements[last_id + 1]
        else:
            check_page_num_e = elements[-1]
        if (
            check_page_num_e.category == ElementType.UNCATEGORIZED_TEXT
            and check_page_num_e.text.strip().isalnum()
        ):
            sidebars_elements.add(check_page_num_e.id)

    elements = [e for e in elements if e.id not in sidebars_elements]
    return elements


def remove_annotation(elements):
    from collections import Counter
    from unstructured.documents.elements import ElementType

    page_num = max(e.metadata.page_number for e in elements)
    un_texts_counter = Counter(
        [e.text for e in elements if e.category == ElementType.UNCATEGORIZED_TEXT]
    )
    rm_text = set()
    for text, count in un_texts_counter.items():
        if count / page_num >= 0.5:
            rm_text.add(text)
    elements = [e for e in elements if e.text not in rm_text]
    return elements


def merge_metadatas(metadatas, return_center=False):
    MAX_NUM = 999999999
    if not metadatas:
        return {}
    p1, p2, p3, p4 = (MAX_NUM, MAX_NUM), (MAX_NUM, 0), (0, 0), (0, MAX_NUM)
    for metadata in metadatas:
        p1_, p2_, p3_, p4_ = metadata["coordinates"]["points"]
        p1 = (min(p1[0], p1_[0]), min(p1[1], p1_[1]))
        p2 = (min(p2[0], p2_[0]), max(p2[1], p2_[1]))
        p3 = (max(p3[0], p3_[0]), max(p3[1], p3_[1]))
        p4 = (max(p4[0], p4_[0]), min(p4[1], p4_[1]))
    points = (p1, p2, p3, p4)
    if return_center:
        points = {"x": (p1[0] + p3[0]) / 2, "y": (p1[1] + p3[1]) / 2}
        page_number = metadata["page_number"]
    return {"points": points, "page_number": page_number}


def create_chunk_and_metadatas(page_elements, stride=3, window=10):
    page_elements = remove_sidebars(page_elements)
    datas = []
    for i in range(0, len(page_elements), stride):
        windown_elements = page_elements[i : i + window]
        metadatas = [e.metadata.to_dict() for e in windown_elements]
        chunk = "\n".join([e.text for e in windown_elements])
        datas.append(
            {"txt": chunk, "metadata": merge_metadatas(metadatas, return_center=True)}
        )
    return datas


def get_chunks(elements):
    from collections import defaultdict

    elements = remove_annotation(elements)

    pages_elements = defaultdict(list)
    for element in elements:
        pages_elements[element.metadata.page_number].append(element)

    all_chunks_and_links = sum(
        [
            create_chunk_and_metadatas(page_elements)
            for _, page_elements in pages_elements.items()
        ],
        [],
    )
    return all_chunks_and_links
