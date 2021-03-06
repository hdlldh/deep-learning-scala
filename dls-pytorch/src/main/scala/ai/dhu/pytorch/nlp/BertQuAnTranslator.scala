package ai.dhu.pytorch.nlp

import java.io.IOException
import java.nio.file.Paths
import java.util

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.ndarray.NDList
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}

class BertQuAnTranslator extends Translator[QAInput, String] {

  private var vocabulary: DefaultVocabulary = _
  private var tokenizer: BertTokenizer = _
  private var tokenList: util.List[String] = _

  @throws[IOException]
  override def prepare(ctx: TranslatorContext): Unit = {
    val path = Paths.get("build/pytorch/bert-qa/vocab.txt")
    vocabulary = DefaultVocabulary.builder
      .optMinFrequency(1)
      .addFromTextFile(path)
      .optUnknownToken("[UNK]")
      .build
    tokenizer = new BertTokenizer()

  }

  override def getBatchifier: Batchifier = Batchifier.STACK

  override def processInput(ctx: TranslatorContext, input: QAInput): NDList = {
    val token = tokenizer.encode(input.getQuestion.toLowerCase, input.getParagraph.toLowerCase, 384)
    // get the encoded tokens that would be used in precessOutput
    tokenList = token.getTokens
    // map the tokens(String) to indices(long)

    val manager = ctx.getNDManager
    val indices = tokenList.stream().mapToLong(vocabulary.getIndex).toArray
    val attentionMask = token.getAttentionMask.stream().mapToLong(i => i).toArray
    val tokenType = token.getTokenTypes.stream().mapToLong(i => i).toArray
    val indicesArray = manager.create(indices)
    val attentionMaskArray = manager.create(attentionMask)
    val tokenTypeArray = manager.create(tokenType)

    new NDList(indicesArray, attentionMaskArray, tokenTypeArray)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): String = {
    val startLogits = list.get(0)
    val endLogits = list.get(1)
    val startIdx = startLogits.argMax().getLong().asInstanceOf[Int]
    val endIdx = endLogits.argMax().getLong().asInstanceOf[Int]
    tokenList.subList(startIdx, endIdx + 1).toString
  }

}
