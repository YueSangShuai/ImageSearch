// Tokenizer.cpp
#include "Tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

Tokenizer::Tokenizer(const std::string& vocab_path) {
    LoadVocab(vocab_path);
}

std::vector<int> Tokenizer::Encode(const std::string& text) const {
    std::vector<int> ids;
    ids.push_back(token2id_.at("[CLS]"));

    auto words = BasicTokenize(text);
    for (const auto& word : words) {
        auto sub_tokens = WordPieceTokenize(word);
        for (const auto& token : sub_tokens) {
            auto it = token2id_.find(token);
            ids.push_back(it != token2id_.end() ? it->second : token2id_.at("[UNK]"));
        }
    }

    ids.push_back(token2id_.at("[SEP]"));
    return ids;
}

std::string Tokenizer::Decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        const auto& token = id2token_.at(id);
        if (token == "[CLS]" || token == "[SEP]" || token == "[PAD]") continue;
        if (token.substr(0, 2) == "##") {
            result += token.substr(2);
        } else {
            result += " " + token;
        }
    }
    return result;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
Tokenizer::EncodeBatch(const std::vector<std::string>& texts, int64_t pad_id) const {
    std::vector<std::vector<int64_t>> all_input_ids;
    for (const auto& text : texts) {
        auto ids = Encode(text);
        all_input_ids.emplace_back(ids.begin(), ids.end());
    }
    return PadAndFlatten(all_input_ids, pad_id);
}

void Tokenizer::LoadVocab(const std::string& vocab_path) {
    std::ifstream infile(vocab_path);
    std::string line;
    int index = 0;
    while (std::getline(infile, line)) {
        token2id_[line] = index;
        id2token_[index] = line;
        ++index;
    }
}

std::string Tokenizer::ToLower(const std::string& input) const {
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

std::vector<std::string> Tokenizer::BasicTokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;
    for (char ch : text) {
        if (std::isspace(ch)) {
            if (!current.empty()) {
                tokens.push_back(ToLower(current));
                current.clear();
            }
        } else if (std::ispunct(ch)) {
            if (!current.empty()) {
                tokens.push_back(ToLower(current));
                current.clear();
            }
            tokens.emplace_back(1, ch);
        } else {
            current += ch;
        }
    }
    if (!current.empty()) {
        tokens.push_back(ToLower(current));
    }
    return tokens;
}

std::vector<std::string> Tokenizer::WordPieceTokenize(const std::string& word) const {
    std::vector<std::string> tokens;
    size_t start = 0;
    bool is_bad = false;

    while (start < word.size()) {
        size_t end = word.size();
        std::string cur_substr;
        while (start < end) {
            std::string substr = word.substr(start, end - start);
            if (start > 0) substr = "##" + substr;
            if (token2id_.find(substr) != token2id_.end()) {
                cur_substr = substr;
                break;
            }
            --end;
        }

        if (cur_substr.empty()) {
            is_bad = true;
            break;
        }

        tokens.push_back(cur_substr);
        start = end;
    }

    if (is_bad) return {"[UNK]"};
    return tokens;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
Tokenizer::PadAndFlatten(const std::vector<std::vector<int64_t>>& sequences, int64_t pad_id) const {
    std::vector<int64_t> flat_input_ids;
    std::vector<int64_t> flat_attention_mask;
    std::vector<int64_t> flat_token_type_ids;

    size_t max_len = 0;
    for (const auto& seq : sequences) {
        if (seq.size() > max_len) max_len = seq.size();
    }

    for (const auto& seq : sequences) {
        size_t pad_len = max_len - seq.size();

        flat_input_ids.insert(flat_input_ids.end(), seq.begin(), seq.end());
        flat_input_ids.insert(flat_input_ids.end(), pad_len, pad_id);

        flat_attention_mask.insert(flat_attention_mask.end(), seq.size(), 1);
        flat_attention_mask.insert(flat_attention_mask.end(), pad_len, 0);

        flat_token_type_ids.insert(flat_token_type_ids.end(), max_len, 0);
    }

    return {flat_input_ids, flat_attention_mask, flat_token_type_ids};
}