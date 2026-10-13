import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";

const EPSILON = 1e-5;
const EMBEDDING_DIMENSION = 768;
const NUM_LAYERS = 12;
const NUM_HEADS = 12;
const HEAD_DIM = EMBEDDING_DIMENSION / NUM_HEADS;
const VOCAB_SIZE = 50257;
const CONTEXT_LENGTH = 1024;

type Tensor = Float32Array;

interface TransformerInput {
  qWeights: Tensor;
  kWeights: Tensor;
  vWeights: Tensor;
  qBiases: Tensor;
  kBiases: Tensor;
  vBiases: Tensor;

  l1Weights: Tensor;
  l1Biases: Tensor;
  l2Weights: Tensor;
  l2Biases: Tensor;

  oWeights: Tensor;
  oBiases: Tensor;

  lnAttnWeights: Tensor;
  lnAttnBiases: Tensor;
  lnMlpWeights: Tensor;
  lnMlpBiases: Tensor;
}

interface GptWeights {
  embeddingWeights: Tensor;
  positionalEmbeddingWeights: Tensor;
  transformerWeights: TransformerInput[];
  finalWeights: Tensor;
  finalBiases: Tensor;
}

interface TokenizerModel {
  vocab: Record<string, number>;
  merges: (string | string[])[];
}

type Rng = () => number;


// Utility used at every stage (tokenizer load, weight load, tokenize): aborts the program on a fatal error.
// In: an error message string. Out: never returns (prints to stderr, exits with status 1).
// e.g. fail("failed to open tokenizer.json") -> stderr "failed to open tokenizer.json", exit code 1
function fail(message: string): never {
  console.error(message);
  process.exit(1);
}


const gpt2Tokens: string[] = [];
const gpt2TokenToTokenId = new Map<string, number>();
const merges = new Map<string, number>();
const byteEncoder = new Map<number, string>();
const byteDecoder = new Map<string, number>();

// Tokenizer setup + tokenize step: builds the Map key for a BPE merge pair.
// In: left and right symbol strings. Out: one string "left right".
// e.g. mergeKey("Ġ", "t") -> "Ġ t"
function mergeKey(left: string, right: string): string {
  return `${left} ${right}`;
}


// Tokenizer setup (called by loadVocab): builds the byte <-> printable-char tables used by tokenize/detokenize.
// In: nothing. Out: nothing returned; fills byteEncoder (256 entries) and byteDecoder.
// e.g. after the call: byteEncoder.get(32) === "Ġ", byteEncoder.get(65) === "A", byteDecoder.get("Ċ") === 10
function buildByteEncoder(): void {
  const range = (from: string, to: string): number[] => {
    const out: number[] = [];
    for (let c = from.codePointAt(0)!; c <= to.codePointAt(0)!; c++) out.push(c);
    return out;
  };
  const printable = [...range("!", "~"), ...range("¡", "¬"), ...range("®", "ÿ")];
  const chars = printable.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!printable.includes(b)) {
      printable.push(b);
      chars.push(256 + n);
      n++;
    }
  }
  for (let i = 0; i < printable.length; i++) {
    byteEncoder.set(printable[i], String.fromCodePoint(chars[i]));
  }
  for (const [b, c] of byteEncoder) {
    byteDecoder.set(c, b);
  }
}


// Tokenize step (used by preTokenize): checks if a character counts as a letter.
// In: one character. Out: boolean.
// e.g. isLetter("a") -> true, isLetter("7") -> false
function isLetter(c: string): boolean {
  const o = c.codePointAt(0)!;
  return (("A" <= c && c <= "Z") || ("a" <= c && c <= "z")
          || (0x00C0 <= o && o <= 0x02AF) || (0x0400 <= o && o <= 0x04FF));
}


// Tokenize step (used by preTokenize): checks if a character is a digit.
// In: one character. Out: boolean.
// e.g. isNumber("7") -> true, isNumber("a") -> false
function isNumber(c: string): boolean {
  return "0" <= c && c <= "9";
}


// Tokenize step (used by preTokenize): checks if a character is whitespace.
// In: one character. Out: boolean.
// e.g. isWhitespace("\n") -> true, isWhitespace("x") -> false
function isWhitespace(c: string): boolean {
  return " \t\n\r\f".includes(c);
}


// Tokenize step 1: splits the prompt into word-ish pieces before BPE.
// In: raw prompt string. Out: array of string pieces.
// e.g. "Hello world, it's 2026!" -> ["Hello", " world", ",", " it", "'s", " 2026", "!"]
function preTokenize(text: string): string[] {
  const chars = Array.from(text);
  const startsWith = (prefixes: string[], at: number): boolean =>
    prefixes.some((p) => chars.slice(at, at + p.length).join("") === p);
  const pieces: string[] = [];
  const n = chars.length;
  let i = 0;
  while (i < n) {
    const start = i;
    if (startsWith(["'s", "'t", "'m", "'d"], i)) {
      i += 2;
    } else if (startsWith(["'re", "'ve", "'ll"], i)) {
      i += 3;
    } else if (isLetter(chars[i]) || (chars[i] === " " && i + 1 < n && isLetter(chars[i + 1]))) {
      i++;
      while (i < n && isLetter(chars[i])) i++;
    } else if (isNumber(chars[i]) || (chars[i] === " " && i + 1 < n && isNumber(chars[i + 1]))) {
      i++;
      while (i < n && isNumber(chars[i])) i++;
    } else if (isWhitespace(chars[i])) {
      while (i < n && isWhitespace(chars[i])) i++;
      if (i < n && i > start + 1) i--;
    } else {
      while (i < n && !(isWhitespace(chars[i]) || isLetter(chars[i]) || isNumber(chars[i]))) i++;
      if (i === start) i++;
    }
    pieces.push(chars.slice(start, i).join(""));
  }
  return pieces;
}


// Tokenize step 3: merges one byte-encoded piece into GPT-2 sub-tokens using merge ranks.
// In: one byte-encoded piece string. Out: array of sub-token strings that exist in the vocab.
// e.g. "Ġtokenization" -> ["Ġtoken", "ization"]
function bytePairEncode(token: string): string[] {
  if (!token) return [];
  if (gpt2TokenToTokenId.has(token)) return [token];
  const word = Array.from(token);
  while (word.length > 1) {
    let bestRank: number | undefined = undefined;
    let bestIdx = -1;
    for (let i = 0; i < word.length - 1; i++) {
      const rank = merges.get(mergeKey(word[i], word[i + 1]));
      if (rank !== undefined && (bestRank === undefined || rank < bestRank)) {
        bestRank = rank;
        bestIdx = i;
      }
    }
    if (bestIdx < 0) break;
    word.splice(bestIdx, 2, word[bestIdx] + word[bestIdx + 1]);
  }
  return word;
}


// Detokenize step (output side of generation): looks up the vocab string for a token id.
// In: integer token id. Out: its token string, or "" if out of range.
// e.g. 995 -> "Ġworld", 99999 -> ""
function getTokenFromTokenId(tokenId: number): string {
  if (0 <= tokenId && tokenId < gpt2Tokens.length) return gpt2Tokens[tokenId];
  return "";
}


// Tokenizer setup (called by loadVocab): reads BPE merge rules from tokenizer.json into the merges Map.
// In: the "model" object from tokenizer.json. Out: nothing returned; fills merges with "left right" -> rank.
// e.g. { merges: [["Ġ", "t"], ["Ġ", "a"]] } -> merges = { "Ġ t" => 0, "Ġ a" => 1 }
function parseMerges(model: TokenizerModel): void {
  merges.clear();
  let rank = 0;
  for (const entry of model.merges) {
    if (Array.isArray(entry) && entry.length === 2) {
      merges.set(mergeKey(entry[0], entry[1]), rank);
      rank++;
    } else if (typeof entry === "string" && entry.includes(" ")) {
      const space = entry.indexOf(" ");
      merges.set(mergeKey(entry.slice(0, space), entry.slice(space + 1)), rank);
      rank++;
    }
  }
}


// Startup / tokenizer setup: loads tokenizer.json and fills all tokenizer tables once before generation.
// In: nothing (reads ../weights/tokenizer/tokenizer.json). Out: nothing returned; fills gpt2Tokens, gpt2TokenToTokenId, merges, byte maps.
// e.g. after the call: gpt2Tokens[995] === "Ġworld", gpt2TokenToTokenId.get("Ġworld") === 995, merges.size === 50000
function loadVocab(): void {
  let model: TokenizerModel;
  try {
    model = JSON.parse(readFileSync("../weights/tokenizer/tokenizer.json", "utf-8")).model;
  } catch {
    fail("failed to open tokenizer.json");
  }

  buildByteEncoder();

  gpt2Tokens.length = 0;
  gpt2Tokens.push(...new Array<string>(VOCAB_SIZE).fill(""));
  gpt2TokenToTokenId.clear();
  for (const [token, tokenId] of Object.entries(model.vocab)) {
    if (0 <= tokenId && tokenId < VOCAB_SIZE) {
      gpt2Tokens[tokenId] = token;
      gpt2TokenToTokenId.set(token, tokenId);
    }
  }

  parseMerges(model);
  console.error(`tokenizer loaded: vocab=${gpt2TokenToTokenId.size} merges=${merges.size}`);
}


// Tokenize step (input side of generation): turns the prompt text into token ids for the model.
// In: prompt string. Out: array of token ids.
// e.g. "Hello world" -> [15496, 995]
function tokenize(text: string): number[] {
  const utf8 = new TextEncoder();
  const ids: number[] = [];
  for (const piece of preTokenize(text)) {
    let encoded = "";
    for (const b of utf8.encode(piece)) encoded += byteEncoder.get(b)!;
    for (const token of bytePairEncode(encoded)) {
      const id = gpt2TokenToTokenId.get(token);
      if (id === undefined) fail("unknown token in vocab");
      ids.push(id);
    }
  }
  return ids;
}


// Detokenize step (output side of generation): turns generated token ids back into text to print.
// In: array of token ids. Out: decoded string.
// e.g. [15496, 995] -> "Hello world"
function detokenize(ids: number[]): string {
  const bytes: number[] = [];
  for (const id of ids) {
    for (const c of getTokenFromTokenId(id)) {
      const b = byteDecoder.get(c);
      if (b !== undefined) bytes.push(b);
    }
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}


// Weight loading: parses whitespace-separated floats from a weight file's bytes into a Tensor.
// In: file Buffer, cursor { offset }, optional maxCount. Out: Float32Array of the floats read (cursor advanced past them).
// e.g. Buffer "0.1 0.2 0.3 0.4", { offset: 0 }, 2 -> Float32Array [0.1, 0.2], pos.offset === 7
function readFromFileStream(buf: Buffer, pos: { offset: number }, maxCount = 0): Tensor {
  const isSpace = (b: number): boolean => b === 32 || (9 <= b && b <= 13);
  let values = new Float32Array(maxCount > 0 ? maxCount : 1024);
  let count = 0;
  let i = pos.offset;
  const n = buf.length;
  while (i < n && (maxCount <= 0 || count < maxCount)) {
    while (i < n && isSpace(buf[i])) i++;
    if (i >= n) break;
    const start = i;
    while (i < n && !isSpace(buf[i])) i++;
    if (count === values.length) {
      const grown = new Float32Array(values.length * 2);
      grown.set(values);
      values = grown;
    }
    values[count++] = Number(buf.toString("latin1", start, i));
  }
  pos.offset = i;
  return count === values.length ? values : values.slice(0, count);
}


// Weight loading: reads one whole weight .txt file into a flat Tensor.
// In: path to a weight .txt file. Out: Float32Array of every float in it.
// e.g. "../weights/transformer.ln_f.bias.txt" -> Float32Array(768)
function load(path: string): Tensor {
  let buf: Buffer;
  try {
    buf = readFileSync(path);
  } catch {
    fail(`failed to open ${path}`);
  }
  return readFromFileStream(buf, { offset: 0 });
}


// Weight loading (per layer): splits GPT-2's fused c_attn weights/biases into separate Q, K, V tensors.
// In: a layer object and a path prefix. Out: nothing returned; sets layer.q/k/vWeights (768x768) and q/k/vBiases (768).
// e.g. (layer, "../weights/transformer.h.0") -> layer.qWeights.length === 589824, layer.qBiases.length === 768
function loadFusedQkv(layer: TransformerInput, prefix: string): void {
  const nEmb = EMBEDDING_DIMENSION;
  const nFused = 3 * nEmb;

  const biasPath = prefix + ".attn.c_attn.bias.txt";
  let biasBuf: Buffer;
  try {
    biasBuf = readFileSync(biasPath);
  } catch {
    fail(`failed to open ${biasPath}`);
  }
  const pos = { offset: 0 };
  layer.qBiases = readFromFileStream(biasBuf, pos, nEmb);
  layer.kBiases = readFromFileStream(biasBuf, pos, nEmb);
  layer.vBiases = readFromFileStream(biasBuf, pos, nEmb);

  const fused = load(prefix + ".attn.c_attn.weight.txt");
  layer.qWeights = new Float32Array(nEmb * nEmb);
  layer.kWeights = new Float32Array(nEmb * nEmb);
  layer.vWeights = new Float32Array(nEmb * nEmb);
  for (let row = 0; row < nEmb; row++) {
    const base = row * nFused;
    layer.qWeights.set(fused.subarray(base, base + nEmb), row * nEmb);
    layer.kWeights.set(fused.subarray(base + nEmb, base + 2 * nEmb), row * nEmb);
    layer.vWeights.set(fused.subarray(base + 2 * nEmb, base + nFused), row * nEmb);
  }
}


// Startup / weight loading: loads every GPT-2 tensor (embeddings, 12 layers, final norm) once before generation.
// In: nothing (reads ../weights/*.txt). Out: a fully populated GptWeights.
// e.g. () -> { embeddingWeights: Float32Array(50257*768), positionalEmbeddingWeights: Float32Array(1024*768), transformerWeights: [12 layers], finalWeights: Float32Array(768), finalBiases: Float32Array(768) }
function loadWeights(): GptWeights {
  console.error("loading embedding weights...");
  const embeddingWeights = load("../weights/transformer.wte.weight.txt");
  const positionalEmbeddingWeights = load("../weights/transformer.wpe.weight.txt");

  const transformerWeights: TransformerInput[] = [];
  for (let i = 0; i < NUM_LAYERS; i++) {
    const prefix = `../weights/transformer.h.${i}`;
    console.error(`loading layer ${i}...`);
    const layer = {} as TransformerInput;

    layer.lnAttnWeights = load(prefix + ".ln_1.weight.txt");
    layer.lnAttnBiases = load(prefix + ".ln_1.bias.txt");

    loadFusedQkv(layer, prefix);

    layer.oWeights = load(prefix + ".attn.c_proj.weight.txt");
    layer.oBiases = load(prefix + ".attn.c_proj.bias.txt");

    layer.lnMlpWeights = load(prefix + ".ln_2.weight.txt");
    layer.lnMlpBiases = load(prefix + ".ln_2.bias.txt");

    layer.l1Weights = load(prefix + ".mlp.c_fc.weight.txt");
    layer.l1Biases = load(prefix + ".mlp.c_fc.bias.txt");
    layer.l2Weights = load(prefix + ".mlp.c_proj.weight.txt");
    layer.l2Biases = load(prefix + ".mlp.c_proj.bias.txt");

    transformerWeights.push(layer);
  }

  const finalWeights = load("../weights/transformer.ln_f.weight.txt");
  const finalBiases = load("../weights/transformer.ln_f.bias.txt");
  console.error("weights loaded.");
  return { embeddingWeights, positionalEmbeddingWeights, transformerWeights, finalWeights, finalBiases };
}


// Transformer block, MLP step: the activation applied after the first MLP layer (via forwardPass).
// In: one float. Out: one float.
// e.g. gelu(1) -> ~0.8412, gelu(0) -> 0
function gelu(x: number): number {
  // tanh approximation, same as gpt-2
  const inner = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}


// Transformer block, attention step: multiplies matrices (e.g. Q @ K^T, scores @ V, concat @ oWeights).
// In: flat row-major a (an x am) and b (bn x bm) with am === bn. Out: flat (an x bm).
// e.g. [1,2,3,4] (2x2) @ [5,6,7,8] (2x2) -> [19,22,43,50]
function matMul(a: Tensor, an: number, am: number, b: Tensor, bn: number, bm: number): Tensor {
  if (am !== bn) throw new Error(`matMul shape mismatch: ${an}x${am} @ ${bn}x${bm}`);
  const out = new Float32Array(an * bm);
  // i-k-j order so the inner loop walks b and out row-wise
  for (let i = 0; i < an; i++) {
    for (let k = 0; k < am; k++) {
      const aik = a[i * am + k];
      if (aik === 0) continue;
      for (let j = 0; j < bm; j++) out[i * bm + j] += aik * b[k * bm + j];
    }
  }
  return out;
}


// Transformer block, attention step: transposes K so scores can be computed as Q @ K^T.
// In: flat row-major a (n x m). Out: flat (m x n).
// e.g. [1,2,3,4,5,6] (2x3) -> [1,4,2,5,3,6] (3x2)
function transpose(a: Tensor, n: number, m: number): Tensor {
  const out = new Float32Array(m * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) out[j * n + i] = a[i * m + j];
  }
  return out;
}


// LM head step (logitsForLastToken): scores the last hidden vector against each vocab embedding.
// In: two equal-length vectors. Out: one float.
// e.g. [1,2,3] . [4,5,6] -> 32
function dotProduct(a: Tensor, b: Tensor): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}


// Embedding + transformer residual steps: element-wise add (wte + wpe, x + attn(x), x + mlp(x)).
// In: two equal-length vectors. Out: a new vector of the same length.
// e.g. [1,2] + [3,4] -> [4,6]
function addVectors(a: Tensor, b: Tensor): Tensor {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}


// Transformer block (ln_1, ln_2) and final norm (ln_f): normalizes one token's embedding.
// In: one token vector (768) plus weights and biases of the same length. Out: normalized vector (768).
// e.g. [1,2,3] with weights [1,1,1], biases [0,0,0] -> ~[-1.2247, 0, 1.2247]
function layerNorm(ogEmbeddings: Tensor, weights: Tensor, biases: Tensor): Tensor {
  const n = ogEmbeddings.length;
  const eps = 1e-5;

  // mean
  let sum = 0;
  for (let i = 0; i < n; i++) sum += ogEmbeddings[i];
  const mean = sum / n;

  // variance (biased, divide by n)
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = ogEmbeddings[i] - mean;
    varSum += d * d;
  }
  const inverseStandardDeviation = 1 / Math.sqrt(varSum / n + eps);

  // normalize, then scale and shift
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (ogEmbeddings[i] - mean) * inverseStandardDeviation * weights[i] + biases[i];
  return out;
}


// Attention step (score rows) and sampling step (top-k logits): turns scores into probabilities.
// In: vector of scores (may contain -Infinity). Out: same-length vector of probabilities summing to 1.
// e.g. [1,2,3] -> ~[0.090, 0.245, 0.665]; [0, -Infinity] -> [1, 0]
function softmax(inputs: Tensor): Tensor {
  const n = inputs.length;

  // subtract max for stability
  let maxi = -Infinity;
  for (let i = 0; i < n; i++) maxi = Math.max(maxi, inputs[i]);

  // exponentiate and sum
  const out = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(inputs[i] - maxi);
    sum += out[i];
  }

  // normalize
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}


// Transformer block, MLP step: one linear layer y = x @ W + b for a single token, optionally with gelu.
// In: weights (in x out), biases (out), one input vector (in), useGelu flag. Out: vector (out).
// e.g. W = [1,0,0,1] (2x2), b = [1,1], x = [2,3] -> [3,4]
function forwardPass(weights: Tensor, biases: Tensor, inputs: Tensor, useGelu = false): Tensor {
  const n = inputs.length, m = biases.length;

  // start from the biases, then add x @ W row by row
  const out = new Float32Array(biases);
  for (let i = 0; i < n; i++) {
    const x = inputs[i];
    for (let j = 0; j < m; j++) out[j] += x * weights[i * m + j];
  }

  if (!useGelu) return out;
  for (let j = 0; j < m; j++) out[j] = gelu(out[j]);
  return out;
}


// Transformer block, attention step: computes one head's causal self-attention output.
// In: normed embeddings (numTokens x 768), head index, full Q/K/V weights and biases. Out: this head's output (numTokens x 64).
// e.g. embeddings (5 x 768), headIdx 0 -> Float32Array(5 * 64)
function attention(embeddings: Tensor, numTokens: number, embedDim: number, headDim: number, headIdx: number,
                   qWeights: Tensor, kWeights: Tensor, vWeights: Tensor,
                   qBiases: Tensor, kBiases: Tensor, vBiases: Tensor): Tensor {
  const n = numTokens, colStart = headIdx * headDim;

  // project every token onto this head's slice of columns: (n x embedDim) @ W[:, colStart:colStart+headDim] + b
  function project(weights: Tensor, biases: Tensor): Tensor {
    const out = new Float32Array(n * headDim);
    for (let t = 0; t < n; t++) {
      for (let d = 0; d < headDim; d++) out[t * headDim + d] = biases[colStart + d];
      for (let e = 0; e < embedDim; e++) {
        const x = embeddings[t * embedDim + e];
        const rowBase = e * embedDim + colStart;
        for (let d = 0; d < headDim; d++) out[t * headDim + d] += x * weights[rowBase + d];
      }
    }
    return out;
  }

  const q = project(qWeights, qBiases);
  const k = project(kWeights, kBiases);
  const v = project(vWeights, vBiases);

  // scores = Q @ K^T / sqrt(headDim)
  const scores = matMul(q, n, headDim, transpose(k, n, headDim), headDim, n);
  const scale = 1 / Math.sqrt(headDim);
  for (let i = 0; i < n * n; i++) scores[i] *= scale;

  // causal mask, then softmax each row
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) scores[i * n + j] = -Infinity;
    scores.set(softmax(scores.subarray(i * n, (i + 1) * n)), i * n);
  }

  // weighted sum of values: (n x n) @ (n x headDim)
  return matMul(scores, n, n, v, n, headDim);
}


// Transformer block, attention step: runs all 12 heads, concatenates them, and applies the output projection.
// In: normed embeddings (numTokens x 768) and all attention weights. Out: (numTokens x 768).
// e.g. embeddings (5 x 768) -> Float32Array(5 * 768)
function multiHeadAttention(numTokens: number, embedDim: number, heads: number, headDim: number, embeddings: Tensor,
                            qWeights: Tensor, kWeights: Tensor, vWeights: Tensor,
                            qBiases: Tensor, kBiases: Tensor, vBiases: Tensor,
                            oWeights: Tensor, oBiases: Tensor): Tensor {
  const n = numTokens;

  // run each head and write its (n x headDim) output into its own column slice of concat
  const concat = new Float32Array(n * embedDim);
  for (let h = 0; h < heads; h++) {
    const headOut = attention(embeddings, n, embedDim, headDim, h,
                              qWeights, kWeights, vWeights, qBiases, kBiases, vBiases);
    const colStart = h * headDim;
    for (let t = 0; t < n; t++) {
      for (let d = 0; d < headDim; d++) concat[t * embedDim + colStart + d] = headOut[t * headDim + d];
    }
  }

  // output projection: concat @ oWeights + oBiases
  const out = matMul(concat, n, embedDim, oWeights, embedDim, embedDim);
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < embedDim; j++) out[t * embedDim + j] += oBiases[j];
  }
  return out;
}


// Transformer block, MLP step: feeds every token through 768 -> 3072 (gelu) -> 768.
// In: normed embeddings (numTokens x 768) and the MLP weights/biases. Out: (numTokens x 768).
// e.g. embeddings (5 x 768) -> Float32Array(5 * 768)
function mlp(numTokens: number, dimensions: number, embeddings: Tensor,
             l1Weights: Tensor, l1Biases: Tensor, l2Weights: Tensor, l2Biases: Tensor): Tensor {
  const n = numTokens;
  const out = new Float32Array(n * dimensions);

  // each token on its own: expand to 3072 with gelu, then project back to 768
  for (let t = 0; t < n; t++) {
    const token = embeddings.subarray(t * dimensions, (t + 1) * dimensions);
    const hidden = forwardPass(l1Weights, l1Biases, token, true);
    out.set(forwardPass(l2Weights, l2Biases, hidden), t * dimensions);
  }
  return out;
}


// Transformer block step: runs one full layer (ln_1 -> attention -> residual -> ln_2 -> mlp -> residual).
// In: one layer's weights, numTokens, hidden state (numTokens x 768). Out: new hidden state (numTokens x 768).
// e.g. (layer 0, 5, hidden (5 x 768)) -> Float32Array(5 * 768)
function transformer(layer: TransformerInput, numTokens: number, embeddings: Tensor): Tensor {
  const n = numTokens, d = EMBEDDING_DIMENSION;

  // layerNorm every token's row on its own
  function normRows(x: Tensor, weights: Tensor, biases: Tensor): Tensor {
    const out = new Float32Array(n * d);
    for (let t = 0; t < n; t++) out.set(layerNorm(x.subarray(t * d, (t + 1) * d), weights, biases), t * d);
    return out;
  }

  // attention sub-block with residual: x = x + attn(ln_1(x))
  const normed1 = normRows(embeddings, layer.lnAttnWeights, layer.lnAttnBiases);
  const attnOut = multiHeadAttention(n, d, NUM_HEADS, HEAD_DIM, normed1,
                                     layer.qWeights, layer.kWeights, layer.vWeights,
                                     layer.qBiases, layer.kBiases, layer.vBiases,
                                     layer.oWeights, layer.oBiases);
  const x = addVectors(embeddings, attnOut);

  // mlp sub-block with residual: x = x + mlp(ln_2(x))
  const normed2 = normRows(x, layer.lnMlpWeights, layer.lnMlpBiases);
  const mlpOut = mlp(n, d, normed2, layer.l1Weights, layer.l1Biases, layer.l2Weights, layer.l2Biases);
  return addVectors(x, mlpOut);
}


// Embedding step: turns token ids into input vectors (token embedding + position embedding).
// In: weights and an array of token ids (length <= 1024). Out: (numTokens x 768).
// e.g. [15496, 995] -> Float32Array(2 * 768) = [wte[15496] + wpe[0], wte[995] + wpe[1]]
function getTokenEmbedding(weights: GptWeights, tokenIds: number[], startPos = 0): Tensor {
  const n = tokenIds.length, d = EMBEDDING_DIMENSION;
  if (startPos + n > CONTEXT_LENGTH) fail(`too many tokens: ${startPos + n} > ${CONTEXT_LENGTH}`);
  const out = new Float32Array(n * d);

  // token t: wte[id] + wpe[startPos + t]
  for (let t = 0; t < n; t++) {
    const id = tokenIds[t], pos = startPos + t;
    if (id < 0 || id >= VOCAB_SIZE) fail(`bad token id ${id}`);
    const tokenVec = weights.embeddingWeights.subarray(id * d, (id + 1) * d);
    const posVec = weights.positionalEmbeddingWeights.subarray(pos * d, (pos + 1) * d);
    out.set(addVectors(tokenVec, posVec), t * d);
  }
  return out;
}


// Full forward pass: embedding -> 12 transformer layers -> final ln_f.
// In: weights and an array of token ids. Out: final normed hidden state (numTokens x 768).
// e.g. [15496, 995] -> Float32Array(2 * 768)
function forwardModel(weights: GptWeights, tokenIds: number[]): Tensor {
  const n = tokenIds.length, d = EMBEDDING_DIMENSION;

  let hidden = getTokenEmbedding(weights, tokenIds);
  for (let i = 0; i < NUM_LAYERS; i++) hidden = transformer(weights.transformerWeights[i], n, hidden);

  // final ln_f on every token's row
  const out = new Float32Array(n * d);
  for (let t = 0; t < n; t++) {
    out.set(layerNorm(hidden.subarray(t * d, (t + 1) * d), weights.finalWeights, weights.finalBiases), t * d);
  }
  return out;
}


// LM head step: scores every vocab token as the next token, using the last position's hidden vector.
// In: weights, normed hidden state (numTokens x 768), numTokens. Out: Float32Array of VOCAB_SIZE (50257) logits.
// e.g. hidden (2 x 768), 2 -> Float32Array(50257), one score per token id
function logitsForLastToken(weights: GptWeights, normedHidden: Tensor, numTokens: number): Tensor {
  const d = EMBEDDING_DIMENSION;
  const last = normedHidden.subarray((numTokens - 1) * d, numTokens * d);

  // tied lm head: score against every row of wte
  const logits = new Float32Array(VOCAB_SIZE);
  for (let v = 0; v < VOCAB_SIZE; v++) logits[v] = dotProduct(last, weights.embeddingWeights.subarray(v * d, (v + 1) * d));
  return logits;
}


// Sampling step: picks the next token id from the logits.
// In: logits (50257), k, temperature, rng. Out: one token id.
// e.g. logits [0.1, 5.0, 2.0], k = 1, temperature 0.8 -> 1
function sampleTopK(logits: Tensor, k: number, temperature: number, rng: Rng): number {
  const n = logits.length;
  if (k <= 0 || k > n) k = n;

  // keep the k highest-scoring ids
  const ids = new Array<number>(n);
  for (let i = 0; i < n; i++) ids[i] = i;
  ids.sort((a, b) => logits[b] - logits[a]);

  // temperature-scaled softmax over just those k (clamped so 0 means near-greedy)
  const temp = Math.max(temperature, 1e-5);
  const scaled = new Float32Array(k);
  for (let i = 0; i < k; i++) scaled[i] = logits[ids[i]] / temp;
  const probs = softmax(scaled);

  // walk the cumulative distribution until it passes r
  const r = rng();
  let cum = 0;
  for (let i = 0; i < k; i++) {
    cum += probs[i];
    if (r < cum) return ids[i];
  }
  return ids[k - 1];
}


// One generation step: forward pass -> logits -> sample, producing the next token.
// In: weights, current token ids, topK, temperature, rng. Out: the next token id.
// e.g. [15496, 995] ("Hello world") -> 11 (",")
function gpt(weights: GptWeights, tokenIds: number[], topK: number, temperature: number, rng: Rng): number {
  const hidden = forwardModel(weights, tokenIds);
  const logits = logitsForLastToken(weights, hidden, tokenIds.length);
  return sampleTopK(logits, topK, temperature, rng);
}


// KV cache: every layer's keys and values for the tokens already processed, so each new token
// only runs itself through the model instead of re-running the whole sequence.
// k[layer] / v[layer] are flat (CONTEXT_LENGTH x 768); rows 0 .. length-1 are filled.
interface KvCache {
  k: Tensor[];
  v: Tensor[];
  length: number;
}


// KV cache: makes an empty cache for one generation (about 75MB: 12 layers x K,V x 1024 x 768 floats).
// In: nothing. Out: a KvCache with length 0.
// e.g. newKvCache() -> { k: [12 x Float32Array(1024*768)], v: [...], length: 0 }
function newKvCache(): KvCache {
  const k: Tensor[] = [], v: Tensor[] = [];
  for (let i = 0; i < NUM_LAYERS; i++) {
    k.push(new Float32Array(CONTEXT_LENGTH * EMBEDDING_DIMENSION));
    v.push(new Float32Array(CONTEXT_LENGTH * EMBEDDING_DIMENSION));
  }
  return { k, v, length: 0 };
}


// KV cache, attention step: multi-head attention for only the new tokens, appending their K and V to the cache.
// In: normed embeddings of the new tokens (numNew x 768) at positions startPos.., one layer's cache rows and weights.
// Out: (numNew x 768), the same rows multiHeadAttention would give for these tokens over the full sequence.
// e.g. startPos 5, numNew 1 -> the 6th token attends over cached rows 0..4 plus itself
function multiHeadAttentionCached(numNew: number, startPos: number, embeddings: Tensor, cacheK: Tensor, cacheV: Tensor,
                                  qWeights: Tensor, kWeights: Tensor, vWeights: Tensor,
                                  qBiases: Tensor, kBiases: Tensor, vBiases: Tensor,
                                  oWeights: Tensor, oBiases: Tensor): Tensor {
  const m = numNew, d = EMBEDDING_DIMENSION, hd = HEAD_DIM;
  const scale = 1 / Math.sqrt(hd);

  // full-width q for the new tokens; their k and v go straight into the cache
  const q = new Float32Array(m * d);
  for (let t = 0; t < m; t++) {
    const x = embeddings.subarray(t * d, (t + 1) * d);
    q.set(forwardPass(qWeights, qBiases, x), t * d);
    cacheK.set(forwardPass(kWeights, kBiases, x), (startPos + t) * d);
    cacheV.set(forwardPass(vWeights, vBiases, x), (startPos + t) * d);
  }

  const concat = new Float32Array(m * d);
  const scores = new Float32Array(startPos + m);
  for (let h = 0; h < NUM_HEADS; h++) {
    const colStart = h * hd;
    for (let t = 0; t < m; t++) {
      const pos = startPos + t;
      const qBase = t * d + colStart;

      // scores against every cached key up to and including this token (causal by construction)
      for (let j = 0; j <= pos; j++) {
        const kBase = j * d + colStart;
        let sum = 0;
        for (let e = 0; e < hd; e++) sum += q[qBase + e] * cacheK[kBase + e];
        scores[j] = sum * scale;
      }
      const probs = softmax(scores.subarray(0, pos + 1));

      // weighted sum of cached values into this head's slice
      const outBase = t * d + colStart;
      for (let j = 0; j <= pos; j++) {
        const p = probs[j], vBase = j * d + colStart;
        for (let e = 0; e < hd; e++) concat[outBase + e] += p * cacheV[vBase + e];
      }
    }
  }

  // output projection, one row per new token
  const out = new Float32Array(m * d);
  for (let t = 0; t < m; t++) out.set(forwardPass(oWeights, oBiases, concat.subarray(t * d, (t + 1) * d)), t * d);
  return out;
}


// KV cache, transformer block step: same as transformer, but only for the new tokens, using this layer's cache.
// In: one layer's weights, its cache rows, startPos, numNew, hidden state of the new tokens (numNew x 768).
// Out: new hidden state for those tokens (numNew x 768).
// e.g. (layer 0, k[0], v[0], 5, 1, hidden (1 x 768)) -> Float32Array(768)
function transformerCached(layer: TransformerInput, cacheK: Tensor, cacheV: Tensor,
                           startPos: number, numNew: number, embeddings: Tensor): Tensor {
  const m = numNew, d = EMBEDDING_DIMENSION;

  // layerNorm every token's row on its own
  function normRows(x: Tensor, weights: Tensor, biases: Tensor): Tensor {
    const out = new Float32Array(m * d);
    for (let t = 0; t < m; t++) out.set(layerNorm(x.subarray(t * d, (t + 1) * d), weights, biases), t * d);
    return out;
  }

  // attention sub-block with residual, reading and extending the cache
  const normed1 = normRows(embeddings, layer.lnAttnWeights, layer.lnAttnBiases);
  const attnOut = multiHeadAttentionCached(m, startPos, normed1, cacheK, cacheV,
                                           layer.qWeights, layer.kWeights, layer.vWeights,
                                           layer.qBiases, layer.kBiases, layer.vBiases,
                                           layer.oWeights, layer.oBiases);
  const x = addVectors(embeddings, attnOut);

  // mlp sub-block with residual (no cache needed, tokens don't mix here)
  const normed2 = normRows(x, layer.lnMlpWeights, layer.lnMlpBiases);
  const mlpOut = mlp(m, d, normed2, layer.l1Weights, layer.l1Biases, layer.l2Weights, layer.l2Biases);
  return addVectors(x, mlpOut);
}


// KV cache, full forward pass: runs only the new tokens through all 12 layers, then advances the cache.
// In: weights, the new token ids (the whole prompt first, then one id per step), the cache. Out: normed hidden (numNew x 768).
// e.g. prompt [15496, 995] -> (2 x 768) and cache.length 2; then [11] -> (1 x 768) and cache.length 3
function forwardModelCached(weights: GptWeights, newTokenIds: number[], cache: KvCache): Tensor {
  const m = newTokenIds.length, d = EMBEDDING_DIMENSION, startPos = cache.length;

  let hidden = getTokenEmbedding(weights, newTokenIds, startPos);
  for (let i = 0; i < NUM_LAYERS; i++) {
    hidden = transformerCached(weights.transformerWeights[i], cache.k[i], cache.v[i], startPos, m, hidden);
  }
  cache.length += m;

  // final ln_f on every token's row
  const out = new Float32Array(m * d);
  for (let t = 0; t < m; t++) {
    out.set(layerNorm(hidden.subarray(t * d, (t + 1) * d), weights.finalWeights, weights.finalBiases), t * d);
  }
  return out;
}


// KV cache, one generation step: like gpt, but only feeds the tokens the cache hasn't seen yet.
// In: weights, the new token ids, the cache, topK, temperature, rng. Out: the next token id.
// e.g. ([15496, 995], empty cache) -> 11 (","); then ([11], same cache) -> the token after ","
function gptCached(weights: GptWeights, newTokenIds: number[], cache: KvCache,
                   topK: number, temperature: number, rng: Rng): number {
  const hidden = forwardModelCached(weights, newTokenIds, cache);
  const logits = logitsForLastToken(weights, hidden, newTokenIds.length);
  return sampleTopK(logits, topK, temperature, rng);
}


// Entry point / generation loop: loads everything, then reads prompts and streams generated text.
// In: prompt and token count from stdin. Out: generated text written to stdout.
// e.g. prompt "Hello world", 3 tokens -> prints "Hello world, I'm"
async function main(): Promise<void> {
  const topK = 40;
  const temperature = 0.8;

  loadVocab();
  const weights = loadWeights();
  const rng: Rng = Math.random;

  // read stdin line by line; null means stdin closed
  const rl = createInterface({ input: process.stdin });
  const lines = rl[Symbol.asyncIterator]();
  async function ask(question: string): Promise<string | null> {
    process.stdout.write(question);
    const next = await lines.next();
    return next.done ? null : next.value;
  }

  while (true) {
    const prompt = await ask("prompt (empty to quit): ");
    if (!prompt) break;
    const count = await ask("num tokens: ");
    if (count === null) break;
    const maxNewTokens = parseInt(count, 10);
    if (!(maxNewTokens > 0)) {
      console.error("need a positive token count");
      continue;
    }

    const tokenIds = tokenize(prompt);
    process.stdout.write(detokenize(tokenIds));

    // first step feeds the whole prompt, every later step feeds just the token it sampled
    const cache = newKvCache();
    let pending = tokenIds.slice();
    for (let i = 0; i < maxNewTokens; i++) {
      if (tokenIds.length >= CONTEXT_LENGTH) break;
      const nextId = gptCached(weights, pending, cache, topK, temperature, rng);
      tokenIds.push(nextId);
      process.stdout.write(detokenize([nextId]));
      if (nextId === 50256) break; // <|endoftext|>
      pending = [nextId];
    }
    process.stdout.write("\n\n");
  }
  rl.close();
}


main();
