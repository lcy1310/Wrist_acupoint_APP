
import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations(<DeviceOrientation>[
    DeviceOrientation.portraitUp,
  ]);
  runApp(const WristAcupointApp());
}

const String kModelAssetPath = 'assets/models/best.onnx';
const int kDefaultInputSize = 224;
const int kKeypointCount = 21;
const Duration kInferenceInterval = Duration(milliseconds: 60);
const Duration kStableSpeakCooldown = Duration(seconds: 3);
const Duration kKeepLastResultDuration = Duration(milliseconds: 120);
const bool kMirrorModelInput = false;

const bool kDebugInferenceLogs = true;
const bool kDebugRejectCandidateLogs = false;
const bool kDebugDrawAllKeypoints = true;
const bool kDebugDrawSkeleton = true;
const bool kDebugDrawKeypointLabels = false;
const int kDebugRowPreviewCount = 3;
const int kDebugValuePreviewCount = 24;
const double kCandidateConfidenceThreshold = 0.28;

const double kMinAcceptBoxAreaRatio = 0.006;
const double kMaxAcceptBoxAreaRatio = 0.55;
const double kMinAcceptBoxAspect = 0.25;
const double kMaxAcceptBoxAspect = 4.00;
const double kMinAcceptVisibleKeypoints = 6;
const double kMinAcceptAnchorKeypoints = 1;
const double kMinAcceptSpreadRatio = 0.10;
const double kMaxAcceptSpreadRatio = 2.60;
const double kMinInitialTrackScore = 0.52;
const double kMinContinueTrackScore = 0.45;
const int kMaxTrackMissFrames = 2;
const int kStableCandidateHitsRequired = 1;

const Map<String, String> kAcupointZh = <String, String>{
  'TaiYuan': '太淵',
  'DaLing': '大陵',
  'ShenMen': '神門',
  'YangXi': '陽溪',
  'YangChi': '陽池',
  'YangGu': '陽谷',
};

const Map<String, Color> kAcupointColors = <String, Color>{
  'TaiYuan': Color(0xFF4DA3FF),
  'DaLing': Color(0xFFFFA24A),
  'ShenMen': Color(0xFF53D38A),
  'YangXi': Color(0xFF4DE0E0),
  'YangChi': Color(0xFFB77BFF),
  'YangGu': Color(0xFFFFD166),
};

const List<String> kSurfaceGuideLines = <String>[
  '左手掌：太淵／大陵／神門',
  '左手背：陽谷／陽池／陽溪',
  '右手掌：神門／大陵／太淵',
  '右手背：陽溪／陽池／陽谷',
];

const List<List<int>> kHandConnections = <List<int>>[
  <int>[0, 1],
  <int>[1, 2],
  <int>[2, 3],
  <int>[3, 4],
  <int>[0, 5],
  <int>[5, 6],
  <int>[6, 7],
  <int>[7, 8],
  <int>[0, 9],
  <int>[9, 10],
  <int>[10, 11],
  <int>[11, 12],
  <int>[0, 13],
  <int>[13, 14],
  <int>[14, 15],
  <int>[15, 16],
  <int>[0, 17],
  <int>[17, 18],
  <int>[18, 19],
  <int>[19, 20],
  <int>[5, 9],
  <int>[9, 13],
  <int>[13, 17],
];

enum AppPhase { boot, permissionCheck, modelLoading, ready, detecting, paused, error }

enum SpeechCommandType { start, stop, read, unknown }

class FrameTensor {
  final Float32List tensor;
  final int originalWidth;
  final int originalHeight;
  final double scale;
  final int padX;
  final int padY;
  final int inputSize;
  final bool mirroredX;

  const FrameTensor({
    required this.tensor,
    required this.originalWidth,
    required this.originalHeight,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.inputSize,
    required this.mirroredX,
  });
}

class PoseCandidate {
  final Rect box;
  final List<Offset> keypoints;
  final List<double> confidences;
  final double score;

  const PoseCandidate({
    required this.box,
    required this.keypoints,
    required this.confidences,
    required this.score,
  });
}

class AcupointResult {
  final Rect box;
  final List<Offset> keypoints;
  final List<double> keypointConfidences;
  final Map<String, Offset> acupoints;
  final int frameWidth;
  final int frameHeight;
  final int sensorOrientation;
  final DateTime timestamp;

  const AcupointResult({
    required this.box,
    required this.keypoints,
    required this.keypointConfidences,
    required this.acupoints,
    required this.frameWidth,
    required this.frameHeight,
    required this.sensorOrientation,
    required this.timestamp,
  });
}

extension OffsetMath on Offset {
  Offset operator +(Offset other) => Offset(dx + other.dx, dy + other.dy);
  Offset operator -(Offset other) => Offset(dx - other.dx, dy - other.dy);
  Offset operator *(double factor) => Offset(dx * factor, dy * factor);
  Offset operator /(double factor) => Offset(dx / factor, dy / factor);
}

class PointSmoother {
  final double alpha;
  List<Offset>? _lastPoints;
  Rect? _lastBox;

  PointSmoother({this.alpha = 0.72});

  void reset() {
    _lastPoints = null;
    _lastBox = null;
  }

  List<Offset> smoothPoints(List<Offset> current) {
    if (_lastPoints == null || _lastPoints!.length != current.length) {
      _lastPoints = List<Offset>.from(current);
      return _lastPoints!;
    }

    final out = <Offset>[];
    for (int i = 0; i < current.length; i++) {
      final prev = _lastPoints![i];
      final now = current[i];
      out.add(Offset(
        prev.dx * alpha + now.dx * (1 - alpha),
        prev.dy * alpha + now.dy * (1 - alpha),
      ));
    }
    _lastPoints = out;
    return out;
  }

  Rect smoothRect(Rect current) {
    if (_lastBox == null) {
      _lastBox = current;
      return current;
    }

    final smoothed = Rect.fromLTRB(
      _lastBox!.left * alpha + current.left * (1 - alpha),
      _lastBox!.top * alpha + current.top * (1 - alpha),
      _lastBox!.right * alpha + current.right * (1 - alpha),
      _lastBox!.bottom * alpha + current.bottom * (1 - alpha),
    );
    _lastBox = smoothed;
    return smoothed;
  }
}

String _describeRawOutput(dynamic value, {int depth = 0}) {
  if (value is! List) return value.runtimeType.toString();
  if (value.isEmpty) return 'List(empty)';
  if (depth >= 2) return 'List(len=${value.length})';
  final first = value.first;
  return 'List(len=${value.length}, first=${_describeRawOutput(first, depth: depth + 1)})';
}

String _shapeText(List<int> shape) => shape.isEmpty ? 'unknown' : shape.join('x');

void _logRows(String tag, List<List<double>> rows) {
  debugPrint('$tag rows=${rows.length}');
  for (int i = 0; i < rows.length && i < kDebugRowPreviewCount; i++) {
    final row = rows[i];
    final minV = row.isEmpty ? 0.0 : row.reduce(math.min);
    final maxV = row.isEmpty ? 0.0 : row.reduce(math.max);
    final preview = row.take(kDebugValuePreviewCount).toList(growable: false);
    debugPrint('$tag row[$i] len=${row.length} min=$minV max=$maxV head=$preview');
  }
}

class YuvToRgbPreprocessor {
  static FrameTensor preprocess(
      CameraImage image, {
        int inputSize = kDefaultInputSize,
        bool mirrorX = kMirrorModelInput,
        int sensorOrientation = 90,
      }) {
    final bool needsRotate = sensorOrientation == 90 || sensorOrientation == 270;
    final int rawWidth = image.width;
    final int rawHeight = image.height;

    // 模型看到的邏輯畫面尺寸（旋轉後）
    final int logicalWidth = needsRotate ? rawHeight : rawWidth;
    final int logicalHeight = needsRotate ? rawWidth : rawHeight;

    final double scale = math.min(inputSize / logicalWidth, inputSize / logicalHeight);
    final int resizedW = (logicalWidth * scale).round();
    final int resizedH = (logicalHeight * scale).round();
    final int padX = ((inputSize - resizedW) / 2).floor();
    final int padY = ((inputSize - resizedH) / 2).floor();

    final Float32List tensor = Float32List(inputSize * inputSize * 3);

    final Plane yPlane = image.planes[0];
    final Plane uPlane = image.planes[1];
    final Plane vPlane = image.planes[2];

    final Uint8List yBytes = yPlane.bytes;
    final Uint8List uBytes = uPlane.bytes;
    final Uint8List vBytes = vPlane.bytes;

    final int yRowStride = yPlane.bytesPerRow;
    final int uRowStride = uPlane.bytesPerRow;
    final int vRowStride = vPlane.bytesPerRow;
    final int uPixelStride = uPlane.bytesPerPixel ?? 1;
    final int vPixelStride = vPlane.bytesPerPixel ?? 1;

    final int planeSize = inputSize * inputSize;
    int indexR = 0;
    int indexG = planeSize;
    int indexB = planeSize * 2;

    for (int dy = 0; dy < inputSize; dy++) {
      final bool insideY = dy >= padY && dy < padY + resizedH;
      final int ly = insideY
          ? (((dy - padY) / scale).floor()).clamp(0, logicalHeight - 1)
          : 0;

      for (int dx = 0; dx < inputSize; dx++) {
        if (!insideY || dx < padX || dx >= padX + resizedW) {
          tensor[indexR++] = 0.0;
          tensor[indexG++] = 0.0;
          tensor[indexB++] = 0.0;
          continue;
        }

        int lx = (((dx - padX) / scale).floor()).clamp(0, logicalWidth - 1);
        if (mirrorX) lx = logicalWidth - 1 - lx;

        // 邏輯座標 (lx, ly) → sensor raw 座標 (rawSx, rawSy)
        final int rawSx, rawSy;
        if (sensorOrientation == 90) {
          // 順時針 90°：logical(x,y) → raw(y, rawHeight-1-x)
          rawSx = ly;
          rawSy = rawHeight - 1 - lx;
        } else if (sensorOrientation == 270) {
          // 逆時針 90°：logical(x,y) → raw(rawWidth-1-y, x)
          rawSx = rawWidth - 1 - ly;
          rawSy = lx;
        } else if (sensorOrientation == 180) {
          rawSx = rawWidth - 1 - lx;
          rawSy = rawHeight - 1 - ly;
        } else {
          rawSx = lx;
          rawSy = ly;
        }

        // clamp 保護，防止邊界浮點誤差
        final int safeRawSx = rawSx.clamp(0, rawWidth - 1);
        final int safeRawSy = rawSy.clamp(0, rawHeight - 1);

        final int yIndex = safeRawSy * yRowStride + safeRawSx;
        final int uvIndexU =
            (safeRawSy >> 1) * uRowStride + (safeRawSx >> 1) * uPixelStride;
        final int uvIndexV =
            (safeRawSy >> 1) * vRowStride + (safeRawSx >> 1) * vPixelStride;

        int y = yBytes[yIndex];
        int u = uBytes[uvIndexU];
        int v = vBytes[uvIndexV];

        y = math.max(0, y - 16);
        u -= 128;
        v -= 128;

        int r = (1.164 * y + 1.596 * v).round();
        int g = (1.164 * y - 0.813 * v - 0.391 * u).round();
        int b = (1.164 * y + 2.018 * u).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        tensor[indexR++] = r / 255.0;
        tensor[indexG++] = g / 255.0;
        tensor[indexB++] = b / 255.0;
      }
    }

    return FrameTensor(
      tensor: tensor,
      originalWidth: logicalWidth,
      originalHeight: logicalHeight,
      scale: scale,
      padX: padX,
      padY: padY,
      inputSize: inputSize,
      mirroredX: mirrorX,
    );
  }
}


class TtsService {
  final FlutterTts _tts = FlutterTts();
  bool _ready = false;
  bool _enabled = true;

  bool get ready => _ready;
  bool get enabled => _enabled;

  Future<void> init() async {
    try {
      await _tts.setLanguage('zh-TW');
      await _tts.setSpeechRate(0.45);
      await _tts.setVolume(1.0);
      await _tts.setPitch(1.0);
      await _tts.awaitSpeakCompletion(false);
      _ready = true;
    } catch (_) {
      _ready = false;
    }
  }

  void setEnabled(bool value) {
    _enabled = value;
    if (!value) {
      _tts.stop();
    }
  }

  Future<void> speak(String text, {bool interrupt = false}) async {
    final clean = text.trim();
    if (!_ready || !_enabled || clean.isEmpty) return;
    try {
      if (interrupt) {
        await _tts.stop();
      }
      await _tts.speak(clean);
    } catch (_) {}
  }

  Future<void> stop() async {
    try {
      await _tts.stop();
    } catch (_) {}
  }
}

class SpeechService {
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _initialized = false;
  bool _listening = false;

  bool get initialized => _initialized;
  bool get isListening => _listening;

  Future<bool> init({
    required void Function(String status) onStatus,
    required void Function(dynamic error) onError,
  }) async {
    try {
      _initialized = await _speech.initialize(onStatus: onStatus, onError: onError);
      return _initialized;
    } catch (_) {
      _initialized = false;
      return false;
    }
  }

  Future<void> startListening({
    required void Function(String recognizedWords, bool isFinal) onResult,
    String? localeId,
  }) async {
    if (!_initialized || _listening) return;
    _listening = true;

    await _speech.listen(
      onResult: (result) {
        onResult(result.recognizedWords, result.finalResult);
        if (result.finalResult) {
          _listening = false;
        }
      },
      listenOptions: stt.SpeechListenOptions(
        localeId: localeId,
        listenFor: const Duration(seconds: 10),
        pauseFor: const Duration(seconds: 2),
        partialResults: true,
        cancelOnError: true,
      ),
    );
  }

  Future<void> stop() async {
    if (!_initialized) return;
    try {
      await _speech.stop();
    } catch (_) {}
    _listening = false;
  }
}

class CameraService {
  CameraController? _controller;
  CameraDescription? _backCamera;

  CameraController? get controller => _controller;
  CameraDescription? get selectedCamera => _backCamera;

  Future<void> init() async {
    final cameras = await availableCameras();
    _backCamera = cameras.firstWhere(
      (cam) => cam.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      _backCamera!,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await _controller!.initialize();
  }

  Future<void> startStream(void Function(CameraImage frame) onFrame) async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      throw StateError('CameraController is not initialized');
    }
    if (controller.value.isStreamingImages) return;
    await controller.startImageStream(onFrame);
  }

  Future<void> stopStream() async {
    final controller = _controller;
    if (controller == null) return;
    if (controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
  }

  Future<void> dispose() async {
    final controller = _controller;
    _controller = null;
    if (controller != null) {
      try {
        if (controller.value.isStreamingImages) {
          await controller.stopImageStream();
        }
      } catch (_) {}
      try {
        await controller.dispose();
      } catch (_) {}
    }
  }
}

class LoadedSession {
  final OrtSession session;
  final String inputName;
  final String outputName;
  final List<int> inputShape;
  final List<int> outputShape;

  const LoadedSession({
    required this.session,
    required this.inputName,
    required this.outputName,
    required this.inputShape,
    required this.outputShape,
  });
}


class OnnxPoseService {
  OnnxRuntime? _runtime;
  LoadedSession? _session;
  bool _ready = false;
  bool _debugEnabled = true;
  int _inputSize = kDefaultInputSize;
  PoseCandidate? _pendingStableCandidate;
  int _pendingStableHits = 0;

  final PointSmoother _smoother = PointSmoother(alpha: 0.78);
  PoseCandidate? _trackedCandidate;
  int _trackMissFrames = 0;
  int _frameIndex = 0;

  bool get ready => _ready;
  int get inputSize => _inputSize;
  void setDebugEnabled(bool value) => _debugEnabled = value;

  void resetTracking() {
    _trackedCandidate = null;
    _pendingStableCandidate = null;
    _pendingStableHits = 0;
    _trackMissFrames = 0;
    _smoother.reset();
    _frameIndex = 0;
  }

  Future<void> init() async {
    _runtime = OnnxRuntime();
    if (kDebugInferenceLogs) {
      debugPrint('ONNX init asset=$kModelAssetPath');
    }
    _session = await _createSession(kModelAssetPath);
    _inputSize = _inputSizeFromShape(_session?.inputShape ?? const <int>[]);
    if (kDebugInferenceLogs) {
      debugPrint('ONNX session loaded inputShape=${_session?.inputShape} outputShape=${_session?.outputShape}');
      debugPrint('ONNX resolved inputSize=$_inputSize');
    }
    _ready = true;
  }

  Future<LoadedSession> _createSession(String assetPath) async {
    final session = await _runtime!.createSessionFromAsset(assetPath);

    String inputName = 'images';
    String outputName = '';
    List<int> inputShape = const <int>[1, 3, kDefaultInputSize, kDefaultInputSize];
    List<int> outputShape = const <int>[];

    try {
      final inputInfo = await session.getInputInfo();
      if (inputInfo.isNotEmpty) {
        final map = inputInfo.first;
        inputName = _pickString(map, const <String>['name', 'Name', 'input_name']) ?? inputName;
        final shape = _pickIntList(map, const <String>['shape', 'Shape', 'dims', 'dimensions']);
        if (shape.isNotEmpty) inputShape = shape;
      }
    } catch (_) {}

    try {
      final outputInfo = await session.getOutputInfo();
      if (outputInfo.isNotEmpty) {
        final map = outputInfo.first;
        outputName = _pickString(map, const <String>['name', 'Name', 'output_name']) ?? outputName;
        final shape = _pickIntList(map, const <String>['shape', 'Shape', 'dims', 'dimensions']);
        if (shape.isNotEmpty) outputShape = shape;
      }
    } catch (_) {}

    if (kDebugInferenceLogs) {
      debugPrint('ONNX session IO inputName=$inputName outputName=$outputName inputShape=$inputShape outputShape=$outputShape');
    }

    return LoadedSession(
      session: session,
      inputName: inputName,
      outputName: outputName,
      inputShape: inputShape,
      outputShape: outputShape,
    );
  }

  String? _pickString(Map<dynamic, dynamic> map, List<String> keys) {
    for (final key in keys) {
      final value = map[key];
      if (value != null) return value.toString();
    }
    return null;
  }

  List<int> _pickIntList(Map<dynamic, dynamic> map, List<String> keys) {
    for (final key in keys) {
      final value = map[key];
      if (value is List) {
        final out = <int>[];
        for (final e in value) {
          if (e is int) {
            out.add(e);
          } else if (e is double) {
            out.add(e.toInt());
          } else if (e is String) {
            final parsed = int.tryParse(e);
            if (parsed != null) out.add(parsed);
          }
        }
        if (out.isNotEmpty) return out;
      }
    }
    return const <int>[];
  }

  int _inputSizeFromShape(List<int> shape) {
    if (shape.length >= 4) {
      final int h = shape[2];
      final int w = shape[3];
      if (h > 0 && w > 0) return h < w ? h : w;
    }
    return kDefaultInputSize;
  }

  int? _featureLengthFromShape(List<int> shape) {
    if (shape.isEmpty) return null;
    final dims = shape.where((d) => d > 0).toList(growable: false);
    if (dims.isEmpty) return null;
    final dimsNoBatch = (dims.length >= 3 && dims.first == 1) ? dims.sublist(1) : dims;
    final candidates = dimsNoBatch.where((d) => d >= 6 && d <= 256).toList(growable: false);
    if (candidates.isEmpty) return null;
    candidates.sort();
    return candidates.first;
  }

  List<double> _toDoubleRow(List<dynamic> row) {
    final out = <double>[];
    for (final e in row) {
      if (e is num) {
        out.add(e.toDouble());
      } else if (e is String) {
        final parsed = double.tryParse(e);
        if (parsed != null) out.add(parsed);
      }
    }
    return out;
  }

  List<List<double>> _flattenRows(dynamic raw) {
    final rows = <List<double>>[];

    void walk(dynamic value) {
      if (value is! List || value.isEmpty) return;
      final first = value.first;
      if (first is List) {
        if (first.isNotEmpty && first.first is List) {
          for (final child in value) {
            walk(child);
          }
          return;
        }
        for (final row in value) {
          if (row is List) {
            final converted = _toDoubleRow(row.cast<dynamic>());
            if (converted.isNotEmpty) rows.add(converted);
          }
        }
        return;
      }
      if (value.every((e) => e is num)) {
        final converted = _toDoubleRow(value.cast<dynamic>());
        if (converted.isNotEmpty) rows.add(converted);
      }
    }

    walk(raw);
    return rows;
  }

  List<List<double>> _transpose(List<List<double>> matrix) {
    if (matrix.isEmpty) return const <List<double>>[];
    final int rows = matrix.length;
    final int cols = matrix.first.length;
    final out = <List<double>>[];
    for (int c = 0; c < cols; c++) {
      final row = <double>[];
      for (int r = 0; r < rows; r++) {
        if (c < matrix[r].length) row.add(matrix[r][c]);
      }
      if (row.length == rows) out.add(row);
    }
    return out;
  }

  List<List<double>> _orientMatrix(List<List<double>> matrix, {int? expectedFeatureLen}) {
    if (matrix.isEmpty) return const <List<double>>[];
    final int rowLen = matrix.first.length;
    final int colLen = matrix.length;

    if (expectedFeatureLen != null && expectedFeatureLen > 0) {
      if (rowLen == expectedFeatureLen) return matrix;
      if (colLen == expectedFeatureLen) return _transpose(matrix);
    }

    final bool rowLooksFeature = rowLen >= 6;
    final bool colLooksFeature = colLen >= 6;
    if (rowLooksFeature && !colLooksFeature) return matrix;
    if (colLooksFeature && !rowLooksFeature) return _transpose(matrix);
    return rowLen <= colLen ? matrix : _transpose(matrix);
  }

  List<List<double>> _extractRows(dynamic rawData, {int? expectedFeatureLen}) {
    final rows = _flattenRows(rawData);
    if (rows.isEmpty) return const <List<double>>[];
    return _orientMatrix(rows, expectedFeatureLen: expectedFeatureLen);
  }

  bool _looksNormalized(List<double> values) {
    return values.every((v) => v.abs() <= 2.0);
  }

  double _normalizeConfidence(double value) {
    if (!value.isFinite) return 0.0;
    if (value >= 0.0 && value <= 1.0) return value;
    if (value > -20.0 && value < 20.0) {
      return 1.0 / (1.0 + math.exp(-value));
    }
    return value.clamp(0.0, 1.0);
  }

  Rect _boxFromKeypoints(List<Offset> keypoints, FrameTensor frame) {
    double minX = double.infinity;
    double minY = double.infinity;
    double maxX = double.negativeInfinity;
    double maxY = double.negativeInfinity;

    for (final p in keypoints) {
      minX = math.min(minX, p.dx);
      minY = math.min(minY, p.dy);
      maxX = math.max(maxX, p.dx);
      maxY = math.max(maxY, p.dy);
    }

    if (!minX.isFinite || !minY.isFinite || !maxX.isFinite || !maxY.isFinite) {
      return const Rect.fromLTWH(0, 0, 1, 1);
    }

    final width = math.max(maxX - minX, 1.0);
    final height = math.max(maxY - minY, 1.0);
    final padding = math.max(math.max(width, height) * 0.10, 8.0);

    final left = (minX - padding).clamp(0.0, frame.originalWidth.toDouble());
    final top = (minY - padding).clamp(0.0, frame.originalHeight.toDouble());
    final right = (maxX + padding).clamp(0.0, frame.originalWidth.toDouble());
    final bottom = (maxY + padding).clamp(0.0, frame.originalHeight.toDouble());

    return Rect.fromLTRB(left, top, right, bottom);
  }

  Offset _toOriginalOffset(double x, double y, FrameTensor frame) {
    // The model output is already in the preprocessed input-pixel space.
    // Keep the mapping deterministic and do not guess normalization from a single point.
    double px = x;
    double py = y;

    px = (px - frame.padX) / frame.scale;
    py = (py - frame.padY) / frame.scale;

    if (frame.mirroredX) {
      px = frame.originalWidth.toDouble() - px;
    }

    px = px.clamp(0.0, frame.originalWidth.toDouble());
    py = py.clamp(0.0, frame.originalHeight.toDouble());
    return Offset(px, py);
  }

  PoseCandidate? _buildCandidate(List<double> row, FrameTensor frame, int keypointStart) {
    const int pointCount = kKeypointCount;
    final int needed = keypointStart + pointCount * 3;
    if (row.length < needed) return null;

    final double rowConf = _normalizeConfidence(row[4]);
    if (!rowConf.isFinite || rowConf < kCandidateConfidenceThreshold) return null;

    final keypoints = <Offset>[];
    final confidences = <double>[];
    for (int i = 0; i < pointCount; i++) {
      final int base = keypointStart + i * 3;
      final double kx = row[base];
      final double ky = row[base + 1];
      final double kv = row[base + 2];
      if (!kx.isFinite || !ky.isFinite || !kv.isFinite) return null;
      keypoints.add(_toOriginalOffset(kx, ky, frame));
      confidences.add(_normalizeConfidence(kv));
    }

    final meanKp = confidences.fold<double>(0.0, (a, b) => a + b) / confidences.length;
    if (meanKp < 0.15) return null;
    final score = (rowConf * 0.35 + meanKp * 0.65).clamp(0.0, 1.0);
    final box = _boxFromKeypoints(keypoints, frame);

    if (box.width <= 1 || box.height <= 1) return null;

    final candidate = PoseCandidate(
      box: box,
      keypoints: keypoints,
      confidences: confidences,
      score: score,
    );

    return _filterHandLikeCandidate(candidate, frame);
  }

  PoseCandidate? _parseRow(List<double> row, FrameTensor frame) {
    if (row.length < 5 + kKeypointCount * 3) return null;

    final candidates = <PoseCandidate>[];
    final starts = row.length >= 6 + kKeypointCount * 3 ? const <int>[5, 6] : const <int>[5];
    for (final keypointStart in starts) {
      final candidate = _buildCandidate(row, frame, keypointStart);
      if (candidate != null) candidates.add(candidate);
    }
    if (candidates.isEmpty) return null;
    candidates.sort((a, b) => b.score.compareTo(a.score));
    return candidates.first;
  }

  double _iou(Rect a, Rect b) {
    final left = math.max(a.left, b.left);
    final top = math.max(a.top, b.top);
    final right = math.min(a.right, b.right);
    final bottom = math.min(a.bottom, b.bottom);

    final interW = math.max(0.0, right - left);
    final interH = math.max(0.0, bottom - top);
    final interArea = interW * interH;
    final unionArea = a.width * a.height + b.width * b.height - interArea;
    if (unionArea <= 0) return 0.0;
    return interArea / unionArea;
  }

  List<PoseCandidate> _nmsCandidates(List<PoseCandidate> items, {double iouThreshold = 0.35}) {
    final sorted = List<PoseCandidate>.from(items)..sort((a, b) => b.score.compareTo(a.score));
    final result = <PoseCandidate>[];

    while (sorted.isNotEmpty) {
      final current = sorted.removeAt(0);
      result.add(current);
      sorted.removeWhere((candidate) => _iou(current.box, candidate.box) > iouThreshold);
    }

    return result;
  }


  List<PoseCandidate> _decodeCandidates(List<List<double>> rows, FrameTensor frame, {int? expectedFeatureLen}) {
    final candidates = <PoseCandidate>[];
    for (final row in rows) {
      if (expectedFeatureLen != null && row.length < expectedFeatureLen) continue;
      final parsed = _parseRow(row, frame);
      if (parsed != null) candidates.add(parsed);
    }
    return candidates;
  }


int _visibleKeypointCount(List<double> confidences, {double threshold = 0.25}) {
  return confidences.where((v) => v >= threshold).length;
}

int _anchorKeypointCount(List<double> confidences, {double threshold = 0.30}) {
  const anchors = <int>[0, 1, 5, 9, 13, 17];
  var count = 0;
  for (final idx in anchors) {
    if (idx < confidences.length && confidences[idx] >= threshold) {
      count++;
    }
  }
  return count;
}

double _candidateAreaRatio(PoseCandidate candidate, FrameTensor frame) {
  final frameArea = math.max(frame.originalWidth * frame.originalHeight, 1.0);
  return (candidate.box.width * candidate.box.height) / frameArea;
}

double _handSpreadRatio(PoseCandidate candidate) {
  if (candidate.keypoints.length < kKeypointCount) return 0.0;
  final wrist = candidate.keypoints[0];
  final indexMcp = candidate.keypoints[5];
  final middleMcp = candidate.keypoints[9];
  final ringMcp = candidate.keypoints[13];
  final pinkyMcp = candidate.keypoints[17];
  final palmWidth = math.max((indexMcp - pinkyMcp).distance, 1e-6);
  final avgReach = (indexMcp - wrist).distance +
      (middleMcp - wrist).distance +
      (ringMcp - wrist).distance +
      (pinkyMcp - wrist).distance;
  return (avgReach / 4.0) / math.max(palmWidth, 1.0);
}

double _signedPerpDistance(Offset point, Offset origin, Offset perpUnit) {
  return (point.dx - origin.dx) * perpUnit.dx + (point.dy - origin.dy) * perpUnit.dy;
}

double _fingerExtensionScore(PoseCandidate candidate) {
  if (candidate.keypoints.length < kKeypointCount) return 0.0;
  final wrist = candidate.keypoints[0];
  const pairs = <List<int>>[
    <int>[1, 4],
    <int>[5, 8],
    <int>[9, 12],
    <int>[13, 16],
    <int>[17, 20],
  ];
  double sum = 0.0;
  for (final pair in pairs) {
    final mcp = candidate.keypoints[pair[0]];
    final tip = candidate.keypoints[pair[1]];
    final mcpDist = (mcp - wrist).distance;
    final tipDist = (tip - wrist).distance;
    if (tipDist > mcpDist) {
      sum += math.min(1.0, (tipDist - mcpDist) / math.max(candidate.box.longestSide, 1.0));
    }
  }
  return sum / pairs.length;
}

bool _looksHandLike(PoseCandidate candidate, FrameTensor frame, {StringBuffer? why}) {
  final areaRatio = _candidateAreaRatio(candidate, frame);
  final aspect = candidate.box.width / math.max(candidate.box.height, 1.0);
  final visible = _visibleKeypointCount(candidate.confidences);
  final anchors = _anchorKeypointCount(candidate.confidences);
  final spread = _handSpreadRatio(candidate);
  final meanKp = _meanKeypointConfidence(candidate);
  final fingerExt = _fingerExtensionScore(candidate);

  if (candidate.keypoints.length < kKeypointCount) {
    why?.write('kpts ');
    return false;
  }

  if (areaRatio < kMinAcceptBoxAreaRatio || areaRatio > kMaxAcceptBoxAreaRatio) {
    why?.write('area=$areaRatio ');
    return false;
  }
  if (aspect < kMinAcceptBoxAspect || aspect > kMaxAcceptBoxAspect) {
    why?.write('aspect=$aspect ');
    return false;
  }
  if (visible < kMinAcceptVisibleKeypoints) {
    why?.write('visible=$visible ');
    return false;
  }
  if (anchors < kMinAcceptAnchorKeypoints) {
    why?.write('anchors=$anchors ');
    return false;
  }
  if (spread < kMinAcceptSpreadRatio || spread > kMaxAcceptSpreadRatio) {
    why?.write('spread=$spread ');
    return false;
  }
  if (meanKp < 0.10) {
    why?.write('meanKp=$meanKp ');
    return false;
  }
  if (fingerExt < 0.04) {
    why?.write('fingerExt=$fingerExt ');
    return false;
  }

  return true;
}

PoseCandidate? _filterHandLikeCandidate(PoseCandidate candidate, FrameTensor frame) {
  final why = StringBuffer();
  if (!_looksHandLike(candidate, frame, why: why)) {
    if (kDebugInferenceLogs && kDebugRejectCandidateLogs) {
      debugPrint(
        'ONNX reject score=${candidate.score.toStringAsFixed(3)} '
        'box=${candidate.box.left.toStringAsFixed(1)},${candidate.box.top.toStringAsFixed(1)},'
        '${candidate.box.width.toStringAsFixed(1)}x${candidate.box.height.toStringAsFixed(1)} '
        'reason=${why.toString().trim()}',
      );
    }
    return null;
  }
  return candidate;
}

double _candidatePriorScore(PoseCandidate candidate, FrameTensor frame, {PoseCandidate? previous}) {
  final frameCenter = Offset(frame.originalWidth / 2.0, frame.originalHeight / 2.0);
  final diag = math.sqrt(
    frame.originalWidth * frame.originalWidth + frame.originalHeight * frame.originalHeight,
  );
  final centerDist = (candidate.box.center - frameCenter).distance / math.max(diag, 1.0);

  final areaRatio = _candidateAreaRatio(candidate, frame);
  final aspect = candidate.box.width / math.max(candidate.box.height, 1.0);
  final kpMean = _meanKeypointConfidence(candidate);
  final visible = _visibleKeypointCount(candidate.confidences);
  final anchors = _anchorKeypointCount(candidate.confidences);
  final spread = _handSpreadRatio(candidate);
  final fingerExt = _fingerExtensionScore(candidate);

  double centerPreference = math.max(0.0, 0.44 - centerDist) * 0.40;
  if (previous != null) {
    final prevCenter = previous.box.center;
    final jump = (candidate.box.center - prevCenter).distance;
    final prevSide = math.max(previous.box.longestSide, 1.0);
    final normalizedJump = jump / prevSide;
    centerPreference -= normalizedJump * 0.22;
  }

  double shapeBonus = 0.0;
  if (areaRatio >= kMinAcceptBoxAreaRatio && areaRatio <= kMaxAcceptBoxAreaRatio) shapeBonus += 0.12;
  if (aspect >= kMinAcceptBoxAspect && aspect <= kMaxAcceptBoxAspect) shapeBonus += 0.08;
  if (spread >= kMinAcceptSpreadRatio && spread <= kMaxAcceptSpreadRatio) shapeBonus += 0.10;
  shapeBonus += (visible / 21.0).clamp(0.0, 1.0) * 0.10;
  shapeBonus += (anchors / 6.0).clamp(0.0, 1.0) * 0.08;
  shapeBonus += kpMean * 0.18;
  shapeBonus += fingerExt * 0.08;

  double sideBonus = 0.0;
  if (candidate.keypoints.length >= kKeypointCount) {
    final wrist = candidate.keypoints[0];
    final thumbCmc = candidate.keypoints[1];
    final indexMcp = candidate.keypoints[5];
    final middleMcp = candidate.keypoints[9];
    final ringMcp = candidate.keypoints[13];
    final pinkyMcp = candidate.keypoints[17];
    final palmCenter = (indexMcp + middleMcp + ringMcp + pinkyMcp) / 4.0;
    final axis = palmCenter - wrist;
    final axisNorm = math.max(axis.distance, 1e-6);
    final axisUnit = axis / axisNorm;
    var perp = Offset(-axisUnit.dy, axisUnit.dx);
    final thumbSide = _signedPerpDistance(thumbCmc, wrist, perp);
    final pinkySide = _signedPerpDistance(pinkyMcp, wrist, perp);
    if (thumbSide > pinkySide) {
      sideBonus = 0.06;
    } else {
      sideBonus = -0.04;
    }
  }

  return candidate.score + shapeBonus + sideBonus + centerPreference - (candidate.score < 0.35 ? 0.04 : 0.0);
}

PoseCandidate? _chooseTrackedCandidate(List<PoseCandidate> candidates, FrameTensor frame) {
  if (candidates.isEmpty) {
    _trackMissFrames++;
    _pendingStableCandidate = null;
    _pendingStableHits = 0;
    if (_trackMissFrames >= kMaxTrackMissFrames) {
      _trackedCandidate = null;
      _smoother.reset();
    }
    return null;
  }

  final sorted = List<PoseCandidate>.from(candidates)
    ..sort((a, b) => b.score.compareTo(a.score));

  final chosen = sorted.first;
  if (chosen.score < kMinContinueTrackScore) {
    _trackMissFrames++;
    if (_trackMissFrames >= kMaxTrackMissFrames) {
      _trackedCandidate = null;
      _pendingStableCandidate = null;
      _pendingStableHits = 0;
      _smoother.reset();
    }
    return null;
  }

  _trackMissFrames = 0;
  _trackedCandidate = chosen;
  _pendingStableCandidate = null;
  _pendingStableHits = 0;
  return chosen;
}

bool _isSameHandRegion(PoseCandidate a, PoseCandidate b) {
  final centerJump = (a.box.center - b.box.center).distance;
  final sizeA = a.box.longestSide;
  final sizeB = b.box.longestSide;
  final sizeDiff = (sizeA - sizeB).abs() / math.max(math.max(sizeA, sizeB), 1.0);
  return centerJump <= math.max(math.max(sizeA, sizeB) * 0.18, 18.0) && sizeDiff <= 0.35;
}

List<Offset> _smoothKeypoints(List<Offset> points) => _smoother.smoothPoints(points);
  Rect _smoothBox(Rect box) => _smoother.smoothRect(box);

  double _meanKeypointConfidence(PoseCandidate candidate) {
    if (candidate.confidences.isEmpty) return 0.0;
    return candidate.confidences.fold<double>(0.0, (a, b) => a + b) / candidate.confidences.length;
  }


  Map<String, Offset> _calcAcupoints(List<Offset> kp) {
    final wrist = kp[0];
    final thumbCmc = kp[1];
    final indexMcp = kp[5];
    final middleMcp = kp[9];
    final ringMcp = kp[13];
    final pinkyMcp = kp[17];

    // Same geometry as live.py.
    final palmCenter = (indexMcp + middleMcp + ringMcp + pinkyMcp) / 4.0;
    final axis = palmCenter - wrist;
    final axisNorm = math.max(axis.distance, 1e-6);
    final axisUnit = axis / axisNorm;

    var perp = Offset(-axisUnit.dy, axisUnit.dx);

    // Same side test as live.py.
    if (_signedPerpDistance(thumbCmc, wrist, perp) < 0) {
      perp = Offset(-perp.dx, -perp.dy);
    }

    final wristLine = wrist - axisUnit * (0.14 * axisNorm);
    final palmWidth = math.max((indexMcp - pinkyMcp).distance, 1e-6);
    final offset = 0.20 * palmWidth;

    final taiyuan = wristLine + perp * (1.5 * offset);
    final daling = wristLine;
    final shenmen = wristLine - perp * (1.0 * offset);

    return <String, Offset>{
      'TaiYuan': taiyuan,
      'DaLing': daling,
      'ShenMen': shenmen,
      'YangXi': taiyuan,
      'YangChi': daling,
      'YangGu': shenmen,
    };
  }

  Future<AcupointResult?> infer(
    CameraImage image, {
    required int sensorOrientation,
    bool Function()? shouldContinue,
  }) async {
    bool alive() => shouldContinue?.call() ?? true;

    if (!_ready || _session == null || !alive()) return null;

    final frame = YuvToRgbPreprocessor.preprocess(
      image,
      inputSize: _inputSize,
      mirrorX: kMirrorModelInput,
      sensorOrientation: sensorOrientation,
    );

    final resolvedShape = _resolveInputShape(_session!.inputShape, frame.inputSize);
    final ortValue = await OrtValue.fromList(frame.tensor, resolvedShape);
    Map<String, OrtValue>? outputs;

    try {
      outputs = await _session!.session.run(<String, OrtValue>{
        _session!.inputName: ortValue,
      });
      if (outputs.isEmpty) return null;

      final outputKey = _session!.outputName.isNotEmpty ? _session!.outputName : outputs.keys.first;
      final value = outputs[outputKey];
      if (value == null) return null;

      if (!alive()) return null;
      final raw = await value.asList();
      final featureLen = _featureLengthFromShape(_session!.outputShape);
      final rows = _extractRows(raw, expectedFeatureLen: featureLen);

      if (kDebugInferenceLogs && _debugEnabled && alive()) {
        debugPrint(
          'ONNX frame#$_frameIndex sensor=$sensorOrientation '
          'input=${_session!.inputName} inputShape=${_session!.inputShape} resolved=$resolvedShape '
          'frameSize=${frame.originalWidth}x${frame.originalHeight} scale=${frame.scale.toStringAsFixed(4)} '
          'pad=(${frame.padX},${frame.padY}) mirror=${frame.mirroredX}',
        );
        debugPrint(
          'ONNX outputShape=${_shapeText(_session!.outputShape)} outputName="${_session!.outputName}" '
          'outputsKeys=${outputs.keys.toList()} outputsLen=${outputs.length}',
        );
        debugPrint('ONNX rawType=${raw.runtimeType} summary=${_describeRawOutput(raw)} featureLen=${featureLen ?? -1} rows=${rows.length}');
        if (rows.isNotEmpty) {
          final firstRow = rows.first;
          debugPrint('ONNX firstRowLen=${firstRow.length} head=${firstRow.take(kDebugValuePreviewCount).toList(growable: false)}');
          _logRows('ONNX', rows);
        } else {
          debugPrint('ONNX rows empty after extract');
        }
      }

      if (!alive()) return null;
      final candidates = _decodeCandidates(rows, frame, expectedFeatureLen: featureLen);

      if (kDebugInferenceLogs && _debugEnabled && alive()) {
        debugPrint('ONNX candidates=${candidates.length}');
        if (candidates.isNotEmpty) {
          final topPreview = candidates.take(5).map((c) {
            return 'score=${c.score.toStringAsFixed(3)} box=${c.box.left.toStringAsFixed(1)},${c.box.top.toStringAsFixed(1)},'
                '${c.box.width.toStringAsFixed(1)}x${c.box.height.toStringAsFixed(1)}';
          }).toList();
          debugPrint('ONNX candidatePreview=$topPreview');
        }
      }

      if (!alive()) return null;
      final nms = _nmsCandidates(candidates);
      if (kDebugInferenceLogs && _debugEnabled && alive()) {
        debugPrint('ONNX nms=${nms.length}');
      }
      if (nms.isEmpty) {
        _trackMissFrames++;
        if (_trackMissFrames >= kMaxTrackMissFrames) {
          _trackedCandidate = null;
          _smoother.reset();
        }
        if (kDebugInferenceLogs && _debugEnabled && alive()) {
          debugPrint('ONNX noAcceptedCandidate miss=$_trackMissFrames tracked=${_trackedCandidate != null}');
        }
        return null;
      }

      if (!alive()) return null;
      final bestRaw = _chooseTrackedCandidate(nms, frame);
      if (bestRaw == null) {
        if (kDebugInferenceLogs && _debugEnabled && alive()) {
          debugPrint('ONNX trackRejected miss=$_trackMissFrames tracked=${_trackedCandidate != null}');
        }
        return null;
      }

      if (!alive()) return null;
      final bestKeypoints = bestRaw.keypoints;
      final bestBox = bestRaw.box;
      final acupoints = _calcAcupoints(bestRaw.keypoints);

      if (kDebugInferenceLogs && _debugEnabled && alive()) {
        final preview = bestKeypoints.take(6).map((p) => '(${p.dx.toStringAsFixed(1)},${p.dy.toStringAsFixed(1)})').toList();
        debugPrint(
          'ONNX best selected score=${bestRaw.score.toStringAsFixed(3)} '
          'box=${bestBox.left.toStringAsFixed(1)},${bestBox.top.toStringAsFixed(1)},${bestBox.width.toStringAsFixed(1)}x${bestBox.height.toStringAsFixed(1)} '
          'kpPreview=$preview',
        );
        debugPrint(
          'ONNX acupoints TY=(${acupoints['TaiYuan']!.dx.toStringAsFixed(1)},${acupoints['TaiYuan']!.dy.toStringAsFixed(1)}) '
          'DL=(${acupoints['DaLing']!.dx.toStringAsFixed(1)},${acupoints['DaLing']!.dy.toStringAsFixed(1)}) '
          'SM=(${acupoints['ShenMen']!.dx.toStringAsFixed(1)},${acupoints['ShenMen']!.dy.toStringAsFixed(1)})',
        );
        final frameCenter = Offset(frame.originalWidth / 2.0, frame.originalHeight / 2.0);
        debugPrint(
          'ONNX frameCenter=(${frameCenter.dx.toStringAsFixed(1)},${frameCenter.dy.toStringAsFixed(1)}) '
          'selectedCenter=(${bestBox.center.dx.toStringAsFixed(1)},${bestBox.center.dy.toStringAsFixed(1)}) '
          'selectedArea=${(bestBox.width * bestBox.height).toStringAsFixed(1)} '
          'tracked=${_trackedCandidate != null} miss=$_trackMissFrames',
        );
      }

      _frameIndex++;

      return AcupointResult(
        box: bestBox,
        keypoints: bestKeypoints,
        keypointConfidences: bestRaw.confidences,
        acupoints: acupoints,
        frameWidth: frame.originalWidth,
        frameHeight: frame.originalHeight,
        sensorOrientation: sensorOrientation,
        timestamp: DateTime.now(),
      );
    } finally {
      try {
        await ortValue.dispose();
      } catch (_) {}
      if (outputs != null) {
        for (final output in outputs.values) {
          try {
            await output.dispose();
          } catch (_) {}
        }
      }
    }
  }

  List<int> _resolveInputShape(List<int> shape, int inputSize) {
    if (shape.length < 4) {
      return <int>[1, 3, inputSize, inputSize];
    }
    return <int>[
      shape[0] > 0 ? shape[0] : 1,
      shape[1] > 0 ? shape[1] : 3,
      shape[2] > 0 ? shape[2] : inputSize,
      shape[3] > 0 ? shape[3] : inputSize,
    ];
  }

  Future<void> dispose() async {
    try {
      await _session?.session.close();
    } catch (_) {}
    _session = null;
    _runtime = null;
    _ready = false;
    resetTracking();
  }
}
class AppController extends ChangeNotifier {
  final CameraService cameraService = CameraService();
  final OnnxPoseService onnxService = OnnxPoseService();
  final TtsService ttsService = TtsService();
  final SpeechService speechService = SpeechService();

  AppPhase phase = AppPhase.boot;
  String statusMessage = '初始化中...';
  bool permissionsOk = false;
  bool modelReady = false;
  bool speechReady = false;
  bool speechEnabled = true;
  bool inferenceBusy = false;
  int _detectionSessionToken = 0;

  Timer? _inferenceTimer;
  CameraImage? _latestFrame;
  DateTime _latestFrameAt = DateTime.fromMillisecondsSinceEpoch(0);

  AcupointResult? _currentResult;
  AcupointResult? get currentResult => _currentResult;
  CameraController? get cameraController => cameraService.controller;
  bool get cameraReady => cameraService.controller?.value.isInitialized ?? false;
  bool get isDetecting => phase == AppPhase.detecting;
  bool get canStart => permissionsOk && modelReady && phase != AppPhase.error;

  bool _spokenFirstStable = false;
  String _lastSignature = '';
  int _sameSignatureHits = 0;
  DateTime _lastStableSpeakAt = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastDetectionAt = DateTime.fromMillisecondsSinceEpoch(0);

  Future<void> initialize() async {
    phase = AppPhase.permissionCheck;
    statusMessage = '檢查權限...';
    notifyListeners();

    final permOk = await _requestPermissions();
    permissionsOk = permOk;
    if (!permOk) {
      phase = AppPhase.error;
      statusMessage = '相機或麥克風權限未開啟';
      notifyListeners();
      return;
    }

    phase = AppPhase.modelLoading;
    statusMessage = '載入模型...';
    notifyListeners();

    try {
      await ttsService.init();
      await _initSpeech();
      await onnxService.init();
      modelReady = true;
      phase = AppPhase.ready;
      statusMessage = '模型已載入';
      notifyListeners();
    } catch (e) {
      phase = AppPhase.error;
      statusMessage = '模型載入失敗：$e';
      notifyListeners();
    }
  }

  Future<bool> _requestPermissions() async {
    final camera = await Permission.camera.request();
    final mic = await Permission.microphone.request();
    return camera.isGranted && mic.isGranted;
  }

  Future<void> _initSpeech() async {
    speechReady = await speechService.init(
      onStatus: (status) => debugPrint('speech status: $status'),
      onError: (error) => debugPrint('speech error: $error'),
    );
  }

  void _resetSpeechLocks() {
    _spokenFirstStable = false;
    _lastSignature = '';
    _sameSignatureHits = 0;
    _lastStableSpeakAt = DateTime.fromMillisecondsSinceEpoch(0);
    _lastDetectionAt = DateTime.fromMillisecondsSinceEpoch(0);
  }

  Future<void> toggleDetection() async {
    if (isDetecting) {
      await stopDetection();
    } else {
      await startDetection();
    }
  }

  Future<void> startDetection() async {
    if (!canStart) return;

    _detectionSessionToken++;
    phase = AppPhase.detecting;
    statusMessage = '偵測中...';
    _latestFrame = null;
    _latestFrameAt = DateTime.fromMillisecondsSinceEpoch(0);
    _resetSpeechLocks();
    onnxService.resetTracking();
    onnxService.setDebugEnabled(true);
    notifyListeners();

    try {
      if (!cameraReady) {
        await cameraService.init();
      }
      await cameraService.startStream(_onCameraFrame);
      _startInferenceLoop();
      unawaited(ttsService.speak('請將手腕放入畫面中央'));
      notifyListeners();
    } catch (e) {
      await _stopInferenceLoop();
      phase = AppPhase.error;
      statusMessage = '相機啟動失敗：$e';
      notifyListeners();
    }
  }

  Future<void> stopDetection() async {
    _detectionSessionToken++;
    onnxService.setDebugEnabled(false);
    await _stopInferenceLoop();
    phase = AppPhase.ready;
    statusMessage = '已停止';
    _currentResult = null;
    _latestFrame = null;
    _resetSpeechLocks();
    onnxService.resetTracking();
    notifyListeners();

    await speechService.stop();
    await cameraService.stopStream();
    await ttsService.stop();
  }

  void _startInferenceLoop() {
    _inferenceTimer?.cancel();
    _inferenceTimer = Timer.periodic(kInferenceInterval, (_) {
      unawaited(_processLatestFrame(_detectionSessionToken));
    });
  }

  Future<void> _stopInferenceLoop() async {
    _inferenceTimer?.cancel();
    _inferenceTimer = null;
    _latestFrame = null;
  }

  void _onCameraFrame(CameraImage image) {
    if (phase != AppPhase.detecting) return;
    _latestFrame = image;
    _latestFrameAt = DateTime.now();
  }

  String _signatureOf(AcupointResult result) {
    final names = const <String>['TaiYuan', 'DaLing', 'ShenMen'];
    return names.map((name) {
      final p = result.acupoints[name];
      if (p == null) return '$name:null';
      return '$name:${p.dx.toStringAsFixed(0)}:${p.dy.toStringAsFixed(0)}';
    }).join('|');
  }

  String _spokenResultText(AcupointResult result) => '目前偵測到太淵、大陵、神門三點';

  String manualResultText() {
    if (_currentResult == null) return '目前尚未偵測到穴位，請將手腕放入畫面中央';
    return _spokenResultText(_currentResult!);
  }

  Future<void> speakManualResult() async {
    await ttsService.speak(manualResultText(), interrupt: true);
  }

  SpeechCommandType parseCommand(String text) {
    final s = text.toLowerCase().replaceAll(' ', '');
    if (s.contains('開始') || s.contains('start') || s.contains('啟動')) return SpeechCommandType.start;
    if (s.contains('停止') || s.contains('stop') || s.contains('關閉')) return SpeechCommandType.stop;
    if (s.contains('朗讀') || s.contains('讀取') || s.contains('read')) return SpeechCommandType.read;
    return SpeechCommandType.unknown;
  }

  Future<void> handleVoiceCommandText(String text) async {
    switch (parseCommand(text)) {
      case SpeechCommandType.start:
        await startDetection();
        break;
      case SpeechCommandType.stop:
        await stopDetection();
        break;
      case SpeechCommandType.read:
        await speakManualResult();
        break;
      case SpeechCommandType.unknown:
        await ttsService.speak('指令未識別', interrupt: true);
        break;
    }
  }

  Future<void> toggleSpeechListening() async {
    if (!speechReady) return;
    if (speechService.isListening) {
      await speechService.stop();
      notifyListeners();
      return;
    }

    await speechService.startListening(
      localeId: 'zh-TW',
      onResult: (recognizedWords, isFinal) async {
        if (recognizedWords.isNotEmpty) {
          statusMessage = '語音：$recognizedWords';
          notifyListeners();
        }
        if (recognizedWords.trim().isEmpty) return;
        if (isFinal) {
          await speechService.stop();
          await handleVoiceCommandText(recognizedWords);
          notifyListeners();
        }
      },
    );
    notifyListeners();
  }

  void toggleTts() {
    speechEnabled = !speechEnabled;
    ttsService.setEnabled(speechEnabled);
    notifyListeners();
  }

  Future<void> _processLatestFrame(int sessionToken) async {
    if (phase != AppPhase.detecting || inferenceBusy || sessionToken != _detectionSessionToken) return;
    final image = _latestFrame;
    if (image == null) return;

    _latestFrame = null;
    if (DateTime.now().difference(_latestFrameAt) > const Duration(milliseconds: 250)) {
      return;
    }

    inferenceBusy = true;
    try {
      final result = await onnxService.infer(
        image,
        sensorOrientation: cameraService.controller?.description.sensorOrientation ?? 90,
        shouldContinue: () => phase == AppPhase.detecting && sessionToken == _detectionSessionToken,
      );

      if (sessionToken != _detectionSessionToken || phase != AppPhase.detecting) {
        _currentResult = null;
        _sameSignatureHits = 0;
        _lastSignature = '';
        notifyListeners();
        return;
      }

      if (kDebugInferenceLogs && cameraService.controller?.value.previewSize != null) {
        final ps = cameraService.controller!.value.previewSize!;
        debugPrint('CAM previewSize=${ps.width}x${ps.height} sensorOrientation=${cameraService.controller!.description.sensorOrientation}');
      }

      if (result == null) {
        _currentResult = null;
        _sameSignatureHits = 0;
        _lastSignature = '';
        _lastDetectionAt = DateTime.fromMillisecondsSinceEpoch(0);
        statusMessage = '目前尚未產生可用 keypoints';
        notifyListeners();
        return;
      }

      _lastDetectionAt = DateTime.now();
      _currentResult = result;
      final signature = _signatureOf(result);

      if (signature == _lastSignature) {
        _sameSignatureHits++;
      } else {
        _sameSignatureHits = 1;
        _lastSignature = signature;
      }

      if (_sameSignatureHits >= 2) {
        final canSpeakAgain = DateTime.now().difference(_lastStableSpeakAt) > kStableSpeakCooldown;
        if (!_spokenFirstStable && canSpeakAgain) {
          _spokenFirstStable = true;
          _lastStableSpeakAt = DateTime.now();
          unawaited(ttsService.speak(_spokenResultText(result)));
        }
      }

      notifyListeners();
    } catch (e, stackTrace) {
      debugPrint('❌ 推論失敗：$e');
      debugPrint('$stackTrace');
      phase = AppPhase.error;
      statusMessage = '推論失敗：$e';
      notifyListeners();
    } finally {
      inferenceBusy = false;
    }
  }

  Future<void> disposeAll() async {
    await _stopInferenceLoop();
    await stopDetection();
    await speechService.stop();
    await ttsService.stop();
    await onnxService.dispose();
    await cameraService.dispose();
  }
}

class WristAcupointApp extends StatefulWidget {
  const WristAcupointApp();

  @override
  State<WristAcupointApp> createState() => _WristAcupointAppState();
}

class _WristAcupointAppState extends State<WristAcupointApp> with WidgetsBindingObserver {
  final AppController controller = AppController();
  bool _initialized = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _boot();
  }

  Future<void> _boot() async {
    await controller.initialize();
    controller.addListener(_rebuild);
    if (mounted) setState(() => _initialized = true);
  }

  void _rebuild() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    controller.removeListener(_rebuild);
    unawaited(controller.disposeAll());
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused || state == AppLifecycleState.inactive) {
      unawaited(controller.stopDetection());
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = ThemeData.dark(useMaterial3: true).copyWith(
      colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1B998B), brightness: Brightness.dark),
      scaffoldBackgroundColor: const Color(0xFF0D1117),
    );

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: '中醫手腕穴道辨識助手',
      theme: theme,
      home: _initialized ? HomeScreen(controller: controller) : const _BootScreen(),
    );
  }
}

class _BootScreen extends StatelessWidget {
  const _BootScreen();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: SizedBox(
          width: 220,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('初始化中...'),
            ],
          ),
        ),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  final AppController controller;

  const HomeScreen({required this.controller});

  @override
  Widget build(BuildContext context) {
    final hasCamera = controller.cameraReady;
    final statusText = controller.statusMessage;
    final detecting = controller.isDetecting;
    final result = controller.currentResult;

    return Scaffold(
      appBar: AppBar(
        title: const Text('中醫手腕穴道辨識助手'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            children: <Widget>[
              _StatusPanel(
                status: statusText,
                speech: controller.speechService.isListening ? '語音中' : '待命',
                detecting: detecting ? '偵測中' : '待機',
                modelReady: controller.modelReady ? '模型已載入' : '模型未就緒',
              ),
              const SizedBox(height: 8),
              const _SurfaceHintPanel(),
              const SizedBox(height: 10),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(22),
                  child: Container(
                    color: const Color(0xFF111827),
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        if (!hasCamera || controller.cameraController == null) {
                          return const Center(child: Text('相機尚未啟動'));
                        }

                        final camera = controller.cameraController!;
                        final previewSize = camera.value.previewSize;
                        final aspectRatio = (previewSize != null && previewSize.height > 0)
                            ? previewSize.height / previewSize.width
                            : camera.value.aspectRatio;

                        return Center(
                          child: AspectRatio(
                            aspectRatio: aspectRatio,
                            child: Stack(
                              fit: StackFit.expand,
                              children: <Widget>[
                                CameraPreview(camera),
                                CustomPaint(
                                  painter: AcupointOverlayPainter(
                                    result: result,
                                    showKeypoints: kDebugDrawAllKeypoints,
                                    showSkeleton: kDebugDrawSkeleton,
                                    showKeypointLabels: kDebugDrawKeypointLabels,
                                  ),
                                  child: const SizedBox.expand(),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Row(
                children: <Widget>[
                  Expanded(
                    child: _PrimaryButton(
                      label: detecting ? '停止偵測' : '開始偵測',
                      color: detecting ? const Color(0xFFE74C3C) : const Color(0xFF1B998B),
                      onPressed: (controller.canStart || detecting)
                          ? () {
                              unawaited(controller.toggleDetection());
                            }
                          : null,
                    ),
                  ),
                  const SizedBox(width: 10),
                  _SquareButton(
                    icon: controller.speechService.isListening ? Icons.mic : Icons.mic_none,
                    label: '語音指令',
                    onPressed: controller.speechReady
                        ? () {
                            unawaited(controller.toggleSpeechListening());
                          }
                        : null,
                  ),
                  const SizedBox(width: 10),
                  _SquareButton(
                    icon: Icons.volume_up,
                    label: controller.speechEnabled ? 'TTS開' : 'TTS關',
                    onPressed: () {
                      controller.toggleTts();
                    },
                  ),
                  const SizedBox(width: 10),
                  _SquareButton(
                    icon: Icons.record_voice_over,
                    label: '朗讀穴位',
                    onPressed: () {
                      unawaited(controller.speakManualResult());
                    },
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                '語音指令：開始偵測 / 停止偵測 / 朗讀穴位',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.white70),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StatusPanel extends StatelessWidget {
  final String status;
  final String speech;
  final String detecting;
  final String modelReady;

  const _StatusPanel({
    required this.status,
    required this.speech,
    required this.detecting,
    required this.modelReady,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFF111827),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white12),
      ),
      child: Wrap(
        spacing: 6,
        runSpacing: 4,
        children: <Widget>[
          _Pill(label: status, compact: true),
          _Pill(label: '狀態 $detecting', compact: true),
          _Pill(label: '語音 $speech', compact: true),
          _Pill(label: modelReady, compact: true),
        ],
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  final String label;
  final bool compact;
  const _Pill({required this.label, this.compact = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: compact ? 7 : 10, vertical: compact ? 3 : 5),
      decoration: BoxDecoration(
        color: const Color(0xFF1F2937),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white12),
      ),
      child: Text(
        label,
        style: TextStyle(fontSize: compact ? 9 : 12),
        overflow: TextOverflow.ellipsis,
      ),
    );
  }
}

class _SurfaceHintPanel extends StatelessWidget {
  const _SurfaceHintPanel();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF111827),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          const Text(
            '左右手與掌背對應',
            style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          for (final line in kSurfaceGuideLines)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Text(
                line,
                style: const TextStyle(fontSize: 11, color: Colors.white70, height: 1.2),
              ),
            ),
          const SizedBox(height: 2),
          const Text(
            '畫面會暫時顯示 21 點與骨架，方便你檢查座標映射。',
            style: TextStyle(fontSize: 10, color: Colors.white54, height: 1.2),
          ),
        ],
      ),
    );
  }
}

class _OverlayDebugCache {
  static String _last = '';

  static bool shouldPrint(String signature) {
    if (signature == _last) return false;
    _last = signature;
    return true;
  }
}

class _PrimaryButton extends StatelessWidget {
  final String label;
  final Color color;
  final VoidCallback? onPressed;

  const _PrimaryButton({
    required this.label,
    required this.color,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 52,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          padding: EdgeInsets.zero,
        ),
        onPressed: onPressed,
        child: Center(
          child: Text(
            label,
            style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
            textAlign: TextAlign.center,
            maxLines: 1,
          ),
        ),
      ),
    );
  }
}

class _SquareButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onPressed;

  const _SquareButton({
    required this.icon,
    required this.label,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 80,
      height: 52,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF243244),
          foregroundColor: Colors.white,
          padding: EdgeInsets.zero,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
        onPressed: onPressed,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Icon(icon, size: 18),
            const SizedBox(height: 2),
            Text(label, style: const TextStyle(fontSize: 11)),
          ],
        ),
      ),
    );
  }
}

class AcupointOverlayPainter extends CustomPainter {
  final AcupointResult? result;
  final bool showKeypoints;
  final bool showSkeleton;
  final bool showKeypointLabels;

  AcupointOverlayPainter({
    required this.result,
    required this.showKeypoints,
    required this.showSkeleton,
    required this.showKeypointLabels,
  });


  @override
  void paint(Canvas canvas, Size size) {
    final result = this.result;
    if (result == null) return;

    final logicalSource = Size(
      math.max(result.frameWidth.toDouble(), 1.0),
      math.max(result.frameHeight.toDouble(), 1.0),
    );

    // orientation 只保留供 debug log 使用
    final orientation = result.sensorOrientation % 360;

    final fitted = applyBoxFit(BoxFit.cover, logicalSource, size);
    final Rect destination = Alignment.center.inscribe(fitted.destination, Offset.zero & size);

    if (kDebugInferenceLogs) {
      final preview = result.keypoints.isNotEmpty ? result.keypoints.first : Offset.zero;
    }

    // 座標已在邏輯空間（preprocess 時已旋轉），直接縮放到螢幕即可
    Offset mapPoint(Offset raw) {
      final sx = destination.width / logicalSource.width;
      final sy = destination.height / logicalSource.height;
      return Offset(
        destination.left + raw.dx * sx,
        destination.top + raw.dy * sy,
      );
    }

    final points = result.keypoints.map(mapPoint).toList(growable: false);

    if (showSkeleton) {
      final paint = Paint()
        ..color = Colors.white.withValues(alpha: 0.30)
        ..strokeWidth = 1.6
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round;

      for (final pair in kHandConnections) {
        final a = pair[0];
        final b = pair[1];
        if (a < points.length && b < points.length) {
          canvas.drawLine(points[a], points[b], paint);
        }
      }
    }

    if (showKeypoints) {
      for (int i = 0; i < points.length; i++) {
        final p = points[i];
        final isRef = <int>{0, 1, 5, 9, 13, 17}.contains(i);
        final radius = isRef ? 4.8 : 3.3;

        canvas.drawCircle(
          p,
          radius + 1.7,
          Paint()..color = Colors.black.withValues(alpha: 0.35),
        );
        canvas.drawCircle(
          p,
          radius,
          Paint()..color = Colors.white.withValues(alpha: 0.82),
        );

        if (showKeypointLabels) {
          final tp = TextPainter(
            text: TextSpan(
              text: '$i',
              style: TextStyle(
                color: isRef ? Colors.amberAccent : Colors.white70,
                fontSize: 9,
                fontWeight: FontWeight.w600,
              ),
            ),
            textDirection: TextDirection.ltr,
          )..layout();
          tp.paint(canvas, p + const Offset(4, -12));
        }
      }
    }

    for (final name in <String>['TaiYuan', 'DaLing', 'ShenMen']) {
      final point = result.acupoints[name];
      if (point == null) continue;
      final p = mapPoint(point);
      canvas.drawCircle(
        p,
        10,
        Paint()..color = Colors.black.withValues(alpha: 0.35),
      );
      canvas.drawCircle(
        p,
        6.6,
        Paint()..color = const Color(0xFFE74C3C),
      );
      canvas.drawCircle(
        p,
        2.8,
        Paint()..color = Colors.white.withValues(alpha: 0.92),
      );
    }

    if (showKeypointLabels) {
      for (final name in <String>['TaiYuan', 'DaLing', 'ShenMen']) {
        final p = result.acupoints[name];
        if (p == null) continue;
        final mp = mapPoint(p);
        final label = kAcupointZh[name] ?? name;
        final tp = TextPainter(
          text: TextSpan(
            text: label,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 10,
              fontWeight: FontWeight.w700,
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        tp.paint(canvas, mp + const Offset(8, -18));
      }
    }
  }
  @override
  bool shouldRepaint(covariant AcupointOverlayPainter oldDelegate) {
    return oldDelegate.result != result ||
        oldDelegate.showKeypoints != showKeypoints ||
        oldDelegate.showSkeleton != showSkeleton ||
        oldDelegate.showKeypointLabels != showKeypointLabels;
  }
}
