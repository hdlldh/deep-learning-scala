package ai.dhu.tensorflow_nlp

import ai.djl.training.util.{DownloadUtils, ProgressBar}
import ai.djl.util.ZipUtils

import java.nio.file.{Files, Paths}

object BertRankDownloader {

  def main(args: Array[String]): Unit = {
    val modelUrl = "https://resources.djl.ai/demo/tensorflow/amazon_review_rank_classification.zip"
    DownloadUtils.download(modelUrl, "build/tensorflow/amazon_review_rank_classification.zip", new ProgressBar)
    val zipFile = Paths.get("build/tensorflow/bert_review_rank_classification.zip")

    val modelDir = Paths.get("build/tensorflow/bert_rank")
    if (Files.notExists(modelDir)) ZipUtils.unzip(Files.newInputStream(zipFile), modelDir)
  }
}
