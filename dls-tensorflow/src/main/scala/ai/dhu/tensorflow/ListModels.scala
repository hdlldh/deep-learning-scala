package ai.dhu.tensorflow

import scala.jdk.CollectionConverters._

import ai.djl.repository.zoo.ModelZoo
import org.slf4j.LoggerFactory

object ListModels {

  val logger = LoggerFactory.getLogger(classOf[ListModels.type])

  def main(args: Array[String]): Unit = {
    ModelZoo.listModels().asScala.foreach { case (app, list) =>
      println(app)
      list.forEach(artifact => println(s"${artifact.getName} $artifact ${artifact.getResourceUri}"))
    }
  }
}
