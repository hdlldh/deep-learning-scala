import java.io.PrintWriter
import java.nio.file.Paths

object JsonParser {

  def main(args: Array[String]): Unit = {

    val inputJsonFile = "build/huggingface/zero-shot-classification/pytorch/bart-large-mnli/vocab.json"
    val outputTextFile = "build/huggingface/zero-shot-classification/pytorch/bart-large-mnli/vocab.txt"
    val url = Paths.get(inputJsonFile).toUri.toURL
    val jsonStr = scala.io.Source.fromFile(url.getPath).mkString
    val vocabList = ujson.read(jsonStr).obj.keySet.toSeq
    val writer = new PrintWriter(outputTextFile)
    vocabList.foreach(r => writer.println(r))
    writer.close()
  }
}