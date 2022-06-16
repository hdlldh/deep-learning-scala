package ai.dhu.pytorch.nlp

import java.nio.file.Paths

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.ndarray.NDList
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}

class HFBartZeroShotClsTranslator extends Translator[QAInput, Double] {

  private var vocabulary: DefaultVocabulary = _
  private var tokenizer: HuggingFaceTokenizer = _
//  private var tokenList: Array[String] = _
  private final val EosToken = "</s>"
  private final val SepToken = "</s>"
  private final val UnkToken = "<unk>"
  private final val TopK = 5

  override def prepare(ctx: TranslatorContext): Unit = {
    val path = Paths.get("build/huggingface/zero-shot-classification/pytorch/bart-large-mnli/vocab.txt")
    vocabulary = DefaultVocabulary.builder
      .optMinFrequency(1)
      .addFromTextFile(path)
      .optUnknownToken(UnkToken)
      .build
    tokenizer = HuggingFaceTokenizer.newInstance("facebook/bart-large-mnli")
  }

  override def processInput(ctx: TranslatorContext, input: QAInput): NDList = {
    val encoded = tokenizer.encode(Array(input.getQuestion, EosToken, SepToken, input.getParagraph))
    // get the encoded tokens that would be used in precessOutput
//    tokenList = encoded.getTokens
    // map the tokens(String) to indices(long)

    val manager = ctx.getNDManager
    val indices = encoded.getIds
    val attentionMask = encoded.getAttentionMask.map(i => i)
    val indicesArray = manager.create(indices)
    val attentionMaskArray = manager.create(attentionMask)

    new NDList(indicesArray)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Double= {
    val ndArray = list.get(0)
    val num1 = math.exp(ndArray.getFloat(0))
    val num2 = math.exp(ndArray.getFloat(2))
    num2 / (num1 + num2)

  }

  override def getBatchifier: Batchifier = Batchifier.STACK
}
