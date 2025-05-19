#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <tuple>

class Tokenizer {
public:
    // 构造函数：加载词表
    explicit Tokenizer(const std::string& vocab_path);

    // 编码：将文本转为 token id 序列
    std::vector<int> Encode(const std::string& text) const;

    // 解码：将 token id 序列还原为文本
    std::string Decode(const std::vector<int>& ids) const;

    // 批量编码：编码多个文本并进行 padding
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
    EncodeBatch(const std::vector<std::string>& texts, int64_t pad_id = 0) const;

private:
    std::unordered_map<std::string, int> token2id_;
    std::unordered_map<int, std::string> id2token_;

    // 加载词表
    void LoadVocab(const std::string& vocab_path);

    // 小写转换
    std::string ToLower(const std::string& input) const;

    // 基本分词器（按空格和标点拆分）
    std::vector<std::string> BasicTokenize(const std::string& text) const;

    // WordPiece 分词器（按词片拆分）
    std::vector<std::string> WordPieceTokenize(const std::string& word) const;

    // padding 并展平为一维向量
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
    PadAndFlatten(const std::vector<std::vector<int64_t>>& sequences, int64_t pad_id) const;
};
