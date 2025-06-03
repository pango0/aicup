import json
import re

pred_json_path = "validation_output.json"
valid_json_path = "../training_2.json"
output_txt_path = "task2_answer.txt"

ALLOWED_LABELS = {
    "PATIENT", "DOCTOR", "USERNAME", "FAMILYNAME", "PERSONALNAME",
    "PROFESSION", "ROOM", "DEPARTMENT", "HOSPITAL", "ORGANIZATION", "STREET", "CITY",
    "DISTRICT", "COUNTY", "STATE", "COUNTRY", "ZIP", "LOCATION-OTHER",
    "AGE", "DATE", "TIME", "DURATION", "SET",
    "PHONE", "FAX", "EMAIL", "URL", "IPADDRESS",
    "SOCIAL_SECURITY_NUMBER", "MEDICAL_RECORD_NUMBER", "HEALTH_PLAN_NUMBER", "ACCOUNT_NUMBER",
    "LICENSE_NUMBER", "VEHICLE_ID", "DEVICE_ID", "BIOMETRIC_ID", "ID_NUMBER"
}

def clean_text(text):
    return text.strip().lower().strip(".,!?;:()[]{}\"'")

def match_exact_span(chunks, target_span):
    target_clean = clean_text(target_span)
    cleaned_chunks = [clean_text(chunk["text"]) for chunk in chunks]

    full_text = " ".join(cleaned_chunks)

    if target_clean not in full_text:
        return None

    span_to_chunk_idx = []
    cursor = 0
    for idx, word in enumerate(cleaned_chunks):
        span_to_chunk_idx.append((cursor, cursor + len(word), idx))
        cursor += len(word) + 1

    start_pos = full_text.index(target_clean)
    end_pos = start_pos + len(target_clean)

    matched_idxs = []
    for s, e, idx in span_to_chunk_idx:
        if not (e <= start_pos or s >= end_pos):
            matched_idxs.append(idx)

    if not matched_idxs:
        return None

    start_chunk = chunks[matched_idxs[0]]
    end_chunk = chunks[matched_idxs[-1]]
    start_time = start_chunk["timestamp"][0]
    end_time = end_chunk["timestamp"][1]

    if start_time is None or end_time is None:
        return None

    return (start_time, end_time)


def clean_text(text):
    return re.sub(r"[^\w\s]", "", text).strip().lower()

with open(pred_json_path, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

with open(valid_json_path, "r", encoding="utf-8") as f:
    validation_data = json.load(f)

with open(output_txt_path, "w", encoding="utf-8") as f:
    for item in pred_data:
        id_ = item["id"]
        try:
            spans = json.loads(item["spans"])
        except json.JSONDecodeError:
            print(f"⚠️ Skipping invalid JSON for id {id_}")
            continue
        for j in validation_data:
            if j["num"] == str(id_):
                chunks = j["chunks"]
                break
        
        seen_spans = set()
        
        # 根據 id 找回該段落的 chunks
        # for span in spans:
        #     for label, target_text in span.items():
        #         matched = False
        #         for chunk in chunks:
        #             if target_text.strip() in chunk["text"].strip():
        #                 start, end = chunk["timestamp"]
        #                 if start is None or end is None:
        #                     print(f"⚠️ Invalid timestamp for chunk '{chunk['text']}' in id {id_}")
        #                     continue
        #                 f.write(f"{id_}\t{label}\t{start:.3f}\t{end:.3f}\t{target_text}\n")
        #                 matched = True
        #                 break
        #         if not matched:
        #             print(f"⚠️ Span '{target_text}' not found in id {id_}")


        # for span in spans:
        #     for label, target_text in span.items():
        #         matched = False
        #         target_clean = clean_text(target_text)

        #         # 對 chunks 做滑動窗口拼接
        #         for i in range(len(chunks)):
        #             combined_text = ""
        #             start_time = None
        #             end_time = None

        #             for j in range(i, min(i + 5, len(chunks))):  # 最多拼接 5 個 chunk
        #                 chunk = chunks[j]
        #                 chunk_text = chunk["text"]
        #                 chunk_clean = clean_text(chunk_text)

        #                 if not combined_text:
        #                     start_time = chunk["timestamp"][0]
        #                 combined_text += " " + chunk_clean
        #                 end_time = chunk["timestamp"][1]

        #                 if end_time is None:  # 有缺 timestamp 就略過
        #                     continue

        #                 if target_clean in combined_text.strip():
        #                     f.write(f"{id_}\t{label}\t{start_time:.3f}\t{end_time:.3f}\t{target_text}\n")
        #                     matched = True
        #                     break

        #             if matched:
        #                 break

        #         if not matched:
        #             print(f"⚠️ Span '{target_text}' not matched in id {id_}")

        # for span in spans:
        #     for label, target_text in span.items():
        #         if label not in ALLOWED_LABELS:
        #             print(f"⚠️ Skipping unknown label: {label}")
        #             continue

        #         key = (label, target_text.strip().lower())
        #         if key in seen_spans:
        #             continue
        #         seen_spans.add(key)

        #         matched = False
        #         target_clean = clean_text(target_text)

        #         for i in range(len(chunks)):
        #             combined_text = ""
        #             start_time = None
        #             end_time = None

        #             for j in range(i, min(i + 5, len(chunks))):  # 最多 5 chunk
        #                 chunk = chunks[j]
        #                 chunk_text = chunk["text"]
        #                 chunk_clean = clean_text(chunk_text)

        #                 if not combined_text:
        #                     start_time = chunk["timestamp"][0]
        #                 combined_text += " " + chunk_clean
        #                 end_time = chunk["timestamp"][1]

        #                 if end_time is None or start_time is None:
        #                     continue

        #                 if target_clean in combined_text.strip():
        #                     f.write(f"{id_}\t{label}\t{start_time:.3f}\t{end_time:.3f}\t{target_text}\n")
        #                     matched = True
        #                     break  # 如果你希望只記一次，可以保留 break；否則註解掉

        #         if not matched:
        #             print(f"⚠️ Span '{target_text}' not matched in id {id_}")
        
        
        for span in spans:
            for label, target_text in span.items():
                if label not in ALLOWED_LABELS:
                    print(f"⚠️ Skipping unknown label: {label}")
                    continue

                key = (label, target_text.strip().lower())
                if key in seen_spans:
                    continue
                seen_spans.add(key)

                match = match_exact_span(chunks, target_text)
                if match:
                    start, end = match
                    f.write(f"{id_}\t{label}\t{start:.3f}\t{end:.3f}\t{target_text}\n")
                else:
                    print(f"⚠️ Span '{target_text}' not matched in id {id_}")
