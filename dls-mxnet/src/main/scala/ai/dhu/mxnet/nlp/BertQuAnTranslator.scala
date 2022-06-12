package ai.dhu.mxnet.nlp

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.translate.{NoBatchifyTranslator, TranslatorContext}

import java.io.IOException
import java.nio.file.Paths
import java.util
import scala.jdk.CollectionConverters._

class BertQuAnTranslator extends NoBatchifyTranslator[QAInput, String] {

  private var vocabulary: DefaultVocabulary = _
  private var tokenizer: BertTokenizer = _
  private var tokenList: util.List[String] = _

  @throws[IOException]
  override def prepare(ctx: TranslatorContext): Unit = {
    val path = Paths.get("build/mxnet/bert-qa/vocab.json").toUri.toURL
    vocabulary = DefaultVocabulary
      .builder
      .optMinFrequency(1)
      .addFromCustomizedFile(path, VocabParser.parseToken)
      .optUnknownToken("[UNK]")
      .build
    tokenizer = new BertTokenizer()
  }

  def toFloatArray(list: util.List[_ <: Number]): Array[Float] = {
    list.asScala.map(r => r.floatValue()).toArray
  }


  override def processInput(ctx: TranslatorContext, input: QAInput): NDList = {
    val token = tokenizer.encode(input.getQuestion.toLowerCase, input.getParagraph.toLowerCase, 384)
    // get the encoded tokens that would be used in precessOutput
    tokenList = token.getTokens
    // map the tokens(String) to indices(long)

    val indices = tokenList.asScala.map(r => new java.lang.Long(vocabulary.getIndex(r))).asJava
    val indexesFloat = toFloatArray(indices)
    val types = toFloatArray(token.getTokenTypes)
    val validLength = token.getValidLength
    val manager = ctx.getNDManager
    val data0 = manager.create(indexesFloat)
    data0.setName("data0")
    val data1 = manager.create(types)
    data1.setName("data1")
    val data2 = manager.create(Array[Float](validLength.toFloat))
    data2.setName("data2")
    new NDList(data0, data1, data2)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): String = {
    val array = list.singletonOrThrow()
    val output = array.split(2, 2)
    // Get the formatted logits result
    val startLogits = output.get(0).reshape(new Shape(1, -1))
    val endLogits = output.get(1).reshape(new Shape(1, -1))
    val startIdx = startLogits.argMax(1).getLong(0).asInstanceOf[Int]
    val endIdx = endLogits.argMax(1).getLong(0).asInstanceOf[Int]
    tokenList.subList(startIdx, endIdx + 1).toString
  }
}
