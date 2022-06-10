
//val scalaTestArtifact = "org.scalatest" %% "scalatest" % "3.2.+" % Test
val slf4jApiArtifact = "org.slf4j" % "slf4j-api" % "1.7.36"
val slf4jSimpleArtifact = "org.slf4j" % "slf4j-simple" % "1.7.36"
val djlApi = "ai.djl" % "api" % "0.17.0"
val djlMxnetEngine = "ai.djl.mxnet" % "mxnet-engine" % "0.17.0"
val djlMxnetModelZoo = "ai.djl.mxnet" % "mxnet-model-zoo" % "0.17.0"
val djlPytorchEngine = "ai.djl.pytorch" % "pytorch-engine" % "0.17.0"
val djlPytorchModelZoo = "ai.djl.pytorch" % "pytorch-model-zoo" % "0.17.0"
val djlTensorflowEngine = "ai.djl.tensorflow" % "tensorflow-engine" % "0.17.0"
val djlTensorflowModelZoo = "ai.djl.tensorflow" % "tensorflow-model-zoo" % "0.17.0"
val protobuf = "com.google.protobuf" % "protobuf-java" % "3.20.1"


lazy val commonSettings = Seq(
  scalacOptions ++= Seq("-deprecation", "-feature", "-Xlint"), // , "-Xfatal-warnings"),
  scalaVersion := "2.13.8",
  libraryDependencies ++= Seq(
    //    scalaTestArtifact,
    slf4jApiArtifact,
    slf4jSimpleArtifact,
    djlApi
  ),
  fork := true,
  organization := "org.meta.dhu",
  assembly / assemblyMergeStrategy := {
    case PathList("org", "apache", "spark", "unused", "UnusedStubClass.class") => MergeStrategy.first
    case PathList(ps @ _*) if ps.last == "module-info.class" => MergeStrategy.first
    case PathList("META-INF", "io.netty.versions.properties") => MergeStrategy.first
    case x =>
      val oldStrategy = (assembly / assemblyMergeStrategy).value
      oldStrategy(x)
  }
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "deep-learning-scala",
    libraryDependencies ++= Seq(
      // Add your dependencies here
    )
  ).
  aggregate(mxnet, pytorch, tensorflow)

lazy val mxnet = (project in file("dls-mxnet")).
  settings(commonSettings: _*).
  settings(
    name := "dls-mxnet",
    libraryDependencies ++= Seq(
      djlMxnetEngine,
      djlMxnetModelZoo
    )
  )

lazy val pytorch = (project in file("dls-pytorch")).
  settings(commonSettings: _*).
  settings(
    name := "dls-pytorch",
    libraryDependencies ++= Seq(
      djlPytorchEngine,
      djlPytorchModelZoo
    )
  )

lazy val tensorflow = (project in file("dls-tensorflow")).
  settings(commonSettings: _*).
  settings(
    name := "dls-tensorflow",
    libraryDependencies ++= Seq(
      djlTensorflowEngine,
      djlTensorflowModelZoo,
      protobuf
    )
  )