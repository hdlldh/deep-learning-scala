package ai.dhu.pytorch.nlp

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.ndarray.NDList
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}

class HFBartZeroShotClsTranslator extends Translator[QAInput, Float] {

  private final val tokenizer = HuggingFaceTokenizer.newInstance("facebook/bart-large-mnli")
  private final val EosToken = "</s>"
  private final val SepToken = "</s>"
  private final val UnkToken = "<unk>"

  override def prepare(ctx: TranslatorContext): Unit = {

  }

  override def processInput(ctx: TranslatorContext, input: QAInput): NDList = {
    val encoded = tokenizer.encode(Array(input.getQuestion, EosToken, SepToken, input.getParagraph))
    val manager = ctx.getNDManager
    val tokenIds = encoded.getIds
    val tokenIdsArray = manager.create(tokenIds)
    new NDList(tokenIdsArray)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Float = {
    val ndArray = list.get(0).toFloatArray
    val num1 = math.exp(ndArray(0)).toFloat
    val num2 = math.exp(ndArray(2)).toFloat
    num2 / (num1 + num2)
  }

  override def getBatchifier: Batchifier = Batchifier.STACK
}
