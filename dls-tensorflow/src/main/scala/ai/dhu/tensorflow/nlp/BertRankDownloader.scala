package ai.dhu.tensorflow.nlp

import ai.djl.training.util.{DownloadUtils, ProgressBar}
import ai.djl.util.ZipUtils

import java.nio.file.{Files, Paths}

object BertRankDownloader {

  def main(args: Array[String]): Unit = {
    val modelUrl = "https://resources.djl.ai/demo/tensorflow/amazon_review_rank_classification.zip"
    DownloadUtils.download(modelUrl, "build/tensorflow/amazon_review_rank_classification.zip", new ProgressBar)
    val zipFile = Paths.get("build/tensorflow/amazon_review_rank_classification.zip")

    val modelDir = Paths.get("build/tensorflow/bert-rank")
    if (Files.notExists(modelDir)) ZipUtils.unzip(Files.newInputStream(zipFile), modelDir)
  }
}
