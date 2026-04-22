package com.github.xandergos.terraindiffusionmc.pipeline;

import ai.onnxruntime.*;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;
import com.github.xandergos.terraindiffusionmc.config.TerrainDiffusionConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Thin wrapper around ONNX Runtime with aggressive VRAM optimization.
 *
 * <p>Only one model is resident in GPU VRAM at a time (GPU-slot swapping).
 * Model weights are kept in CPU RAM between inference calls and uploaded to
 * GPU on demand. This keeps peak VRAM to a single model's footprint instead
 * of all three simultaneously.
 */
public final class OnnxModel implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(OnnxModel.class);
    private static final String OPTIMIZED_MODELS_DIR_NAME = "onnx-cache";

    // GPU slot: when offload_models=true, only one session is alive at a time.
    private static final Object GPU_SLOT_LOCK = new Object();
    private static OnnxModel gpuSlotHolder = null;
    private static OrtSession activeGpuSession = null;

    private final OrtEnvironment env;
    private final byte[] optimizedModelBytes;
    private final String name;
    private OrtSession cpuSession;    // non-null in CPU-only mode
    private OrtSession gpuSession;    // non-null when offload_models=false

    private static final class OptimizedModelLoadResult {
        private final byte[] modelBytes;
        private final Path optimizedModelPath;
        private final boolean loadedFromCache;

        private OptimizedModelLoadResult(byte[] modelBytes, Path optimizedModelPath, boolean loadedFromCache) {
            this.modelBytes = modelBytes;
            this.optimizedModelPath = optimizedModelPath;
            this.loadedFromCache = loadedFromCache;
        }
    }

    public OnnxModel(Path modelFilePath, String name) {
        this.name = name;
        try {
            long start = System.currentTimeMillis();
            this.env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
            byte[] sourceModelBytes = Files.readAllBytes(modelFilePath);
            OptimizedModelLoadResult initialOptimizedModelLoadResult = optimizeModelAtRuntime(sourceModelBytes, false);
            byte[] loadedModelBytes;
            try {
                initializeModelSession(initialOptimizedModelLoadResult.modelBytes, start);
                loadedModelBytes = initialOptimizedModelLoadResult.modelBytes;
            } catch (Exception initialLoadException) {
                if (!initialOptimizedModelLoadResult.loadedFromCache) {
                    throw initialLoadException;
                }
                closeLoadedSessions();
                LOG.warn("Cached optimized ONNX model '{}' failed to load. Rebuilding cache: {}",
                        name, initialLoadException.getMessage());
                deleteOptimizedCacheFile(initialOptimizedModelLoadResult.optimizedModelPath);
                OptimizedModelLoadResult rebuiltOptimizedModelLoadResult = optimizeModelAtRuntime(sourceModelBytes, true);
                initializeModelSession(rebuiltOptimizedModelLoadResult.modelBytes, start);
                loadedModelBytes = rebuiltOptimizedModelLoadResult.modelBytes;
            }
            this.optimizedModelBytes = loadedModelBytes;
        } catch (Exception e) {
            throw new RuntimeException("Failed to load ONNX model: " + modelFilePath, e);
        }
    }

    /**
     * Optimizes model bytes and caches the optimized file in the config directory.
     * Falls back to the source model bytes if optimization or cache I/O fails.
     */
    private OptimizedModelLoadResult optimizeModelAtRuntime(byte[] sourceModelBytes, boolean forceRebuildFromSource) {
        Path optimizedModelPath = resolveOptimizedModelPath(sourceModelBytes);
        try {
            if (!forceRebuildFromSource && Files.exists(optimizedModelPath)) {
                byte[] cachedOptimizedModelBytes = Files.readAllBytes(optimizedModelPath);
                return new OptimizedModelLoadResult(cachedOptimizedModelBytes, optimizedModelPath, true);
            }

            Files.createDirectories(optimizedModelPath.getParent());
            Path temporaryOptimizedModelPath = optimizedModelPath.resolveSibling(optimizedModelPath.getFileName() + ".tmp");
            Files.deleteIfExists(temporaryOptimizedModelPath);
            OrtSession.SessionOptions optimizationOptions = new OrtSession.SessionOptions();
            optimizationOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT);
            optimizationOptions.setOptimizedModelFilePath(temporaryOptimizedModelPath.toAbsolutePath().toString());
            try (OrtSession ignored = env.createSession(sourceModelBytes, optimizationOptions)) {
                // Session creation materializes the optimized model on disk.
            }
            byte[] optimizedModelBytesFromDisk = Files.readAllBytes(temporaryOptimizedModelPath);
            Files.move(
                    temporaryOptimizedModelPath,
                    optimizedModelPath,
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE
            );
            LOG.info("Optimized ONNX model '{}' at runtime ({} KB -> {} KB)",
                    name, sourceModelBytes.length / 1024, optimizedModelBytesFromDisk.length / 1024);
            return new OptimizedModelLoadResult(optimizedModelBytesFromDisk, optimizedModelPath, false);
        } catch (Exception optimizationException) {
            LOG.warn("Runtime ONNX optimization failed for '{}', using source model bytes: {}",
                    name, optimizationException.getMessage());
            return new OptimizedModelLoadResult(sourceModelBytes, optimizedModelPath, false);
        }
    }

    /**
     * Loads model sessions for the active inference device configuration.
     */
    private void initializeModelSession(byte[] modelBytes, long startMillis) throws OrtException {
        if ("cpu".equals(TerrainDiffusionConfig.inferenceDevice())) {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            this.cpuSession = env.createSession(modelBytes, sessionOptions);
            this.gpuSession = null;
            LOG.info("ONNX model '{}' loaded on CPU ({} KB) in {} ms",
                    name, modelBytes.length / 1024, System.currentTimeMillis() - startMillis);
            return;
        }
        if (!TerrainDiffusionConfig.offloadModels()) {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            addGpuProvider(sessionOptions);
            this.gpuSession = env.createSession(modelBytes, sessionOptions);
            this.cpuSession = null;
            LOG.info("ONNX model '{}' loaded on GPU ({} KB) in {} ms",
                    name, modelBytes.length / 1024, System.currentTimeMillis() - startMillis);
            return;
        }
        this.cpuSession = null;
        this.gpuSession = null;
        LOG.info("ONNX model '{}' bytes cached in CPU RAM ({} KB) in {} ms",
                name, modelBytes.length / 1024, System.currentTimeMillis() - startMillis);
    }

    private void closeLoadedSessions() {
        if (cpuSession != null) {
            try { cpuSession.close(); } catch (OrtException ignored) {}
            cpuSession = null;
        }
        if (gpuSession != null) {
            try { gpuSession.close(); } catch (OrtException ignored) {}
            gpuSession = null;
        }
    }

    private void deleteOptimizedCacheFile(Path optimizedModelPath) {
        try {
            Files.deleteIfExists(optimizedModelPath);
        } catch (Exception deleteException) {
            LOG.warn("Failed to delete optimized cache '{}' for '{}': {}",
                    optimizedModelPath, name, deleteException.getMessage());
        }
    }

    /**
     * Resolves a deterministic cache file path for an optimized model.
     */
    private Path resolveOptimizedModelPath(byte[] sourceModelBytes) {
        String sourceModelHashPrefix = sha256Hex(sourceModelBytes).substring(0, 16);
        String runtimeVersionTag = resolveOnnxRuntimeVersionTag();
        String optimizedFileName = name + "-" + runtimeVersionTag + "-" + sourceModelHashPrefix + ".onnx";
        return ModelAssetManager.resolveAssetPath(OPTIMIZED_MODELS_DIR_NAME)
                .resolve(optimizedFileName);
    }

    /**
     * Returns the ONNX Runtime version used as part of the optimization cache key.
     */
    private static String resolveOnnxRuntimeVersionTag() {
        Package onnxRuntimePackage = OrtEnvironment.class.getPackage();
        String implementationVersion = onnxRuntimePackage == null ? null : onnxRuntimePackage.getImplementationVersion();
        return implementationVersion == null ? "unknown" : implementationVersion;
    }

    /**
     * Computes a lowercase SHA-256 hex string for deterministic cache naming.
     */
    private static String sha256Hex(byte[] inputBytes) {
        try {
            MessageDigest messageDigest = MessageDigest.getInstance("SHA-256");
            byte[] digestBytes = messageDigest.digest(inputBytes);
            StringBuilder hexBuilder = new StringBuilder(digestBytes.length * 2);
            for (byte digestByte : digestBytes) {
                hexBuilder.append(String.format("%02x", digestByte));
            }
            return hexBuilder.toString();
        } catch (NoSuchAlgorithmException noSuchAlgorithmException) {
            throw new IllegalStateException("Missing SHA-256 algorithm", noSuchAlgorithmException);
        }
    }

    /**
     * Run the model with a flat float array for each named input.
     * Each entry in {@code inputs} is (name, float[] data, long[] shape).
     *
     * @return the output tensor as a flat float array
     */
    public float[] run(Object[][] inputs) {
        if (cpuSession != null) {
            return runWithSession(cpuSession, inputs);
        }
        if (gpuSession != null) {
            return runWithSession(gpuSession, inputs);
        }
        synchronized (GPU_SLOT_LOCK) {
            claimGpuSlot();
            return runWithSession(activeGpuSession, inputs);
        }
    }

    /** Convenience: run with x, noise_labels, and optional cond tensors. */
    public float[] runModel(float[] x, long[] xShape,
                            float[] noiseLabels,
                            float[][] condInputs, long[][] condShapes) {
        int nCond = condInputs == null ? 0 : condInputs.length;
        Object[][] inputs = new Object[2 + nCond][3];
        inputs[0] = new Object[]{"x", x, xShape};
        inputs[1] = new Object[]{"noise_labels", noiseLabels, new long[]{noiseLabels.length}};
        for (int i = 0; i < nCond; i++)
            inputs[2 + i] = new Object[]{"cond_" + i, condInputs[i], condShapes[i]};
        return run(inputs);
    }

    /**
     * Evicts the current GPU session if this model doesn't hold the slot,
     * then creates a fresh GPU session from CPU-cached weights.
     * Must be called under GPU_SLOT_LOCK.
     */
    private void claimGpuSlot() {
        if (gpuSlotHolder == this) return;

        if (activeGpuSession != null) {
            LOG.debug("Evicting '{}' from GPU, loading '{}'",
                    gpuSlotHolder != null ? gpuSlotHolder.name : "?", name);
            try { activeGpuSession.close(); } catch (OrtException ignored) {}
            activeGpuSession = null;
            gpuSlotHolder = null;
        }

        try {
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            addGpuProvider(opts);
            activeGpuSession = env.createSession(optimizedModelBytes, opts);
            gpuSlotHolder = this;
            LOG.debug("GPU session ready for '{}'", name);
        } catch (OrtException e) {
            throw new RuntimeException("Failed to create GPU session for: " + name, e);
        }
    }

    private static void addGpuProvider(OrtSession.SessionOptions opts) throws OrtException {
        boolean gpuRequired = "gpu".equals(TerrainDiffusionConfig.inferenceDevice());
        boolean added = false;

        try {
            OrtCUDAProviderOptions cudaOpts = new OrtCUDAProviderOptions(0);
            // Only grow the BFC arena by exactly what is needed, never pre-allocate.
            cudaOpts.add("arena_extend_strategy", "kSameAsRequested");
            // Heuristic: fast startup, no exhaustive benchmarking, workspace-efficient.
            cudaOpts.add("cudnn_conv_algo_search", "HEURISTIC");
            cudaOpts.add("do_copy_in_default_stream", "1");
            opts.addCUDA(cudaOpts);
            cudaOpts.close();
            added = true;
            LOG.info("Terrain diffusion inference: GPU (CUDA)");
        } catch (Throwable t) {
            LOG.warn("CUDA not available: {} - {}", t.getClass().getSimpleName(), t.getMessage());
            logCudaLoadDiagnostics(t);
        }

        if (!added) {
            try {
                opts.addDirectML(0);
                added = true;
                LOG.info("Terrain diffusion inference: GPU (DirectML)");
            } catch (Throwable t) {
                LOG.warn("DirectML not available: {} - {}", t.getClass().getSimpleName(), t.getMessage());
            }
        }
        if (gpuRequired && !added) {
            throw new OrtException(
                    "inference.device=gpu but neither CUDA nor DirectML is available. " +
                    "Use the GPU build or set inference.device=cpu.");
        }
        if (!added) {
            LOG.info("Terrain diffusion inference: CPU (fallback)");
            LOG.warn("No GPU provider loaded. Check drivers and that the mod jar is the GPU build.");
        }
    }

    /**
     * Emit actionable diagnostics when the ONNX Runtime CUDA provider fails to load.
     *
     * <p>On Windows the stock ORT error ("LoadLibrary failed with error 126") does not
     * indicate which dependency is missing. This method walks the current {@code PATH}
     * entries, reports which CUDA / cuDNN DLLs are resolvable, flags version conflicts
     * (multiple {@code cudart64_*.dll} on PATH), and checks the cuDNN 9 split DLLs that
     * are commonly missing when only {@code cudnn64_9.dll} is present. On non-Windows
     * hosts a short pointer to the ORT install guide is logged instead.
     */
    private static void logCudaLoadDiagnostics(Throwable originalFailure) {
        try {
            String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
            boolean isWindows = osName.contains("win");
            LOG.warn("CUDA provider diagnostics (os='{}', arch='{}', java.vm='{}')",
                    System.getProperty("os.name"),
                    System.getProperty("os.arch"),
                    System.getProperty("java.vm.name"));

            String rawFailureMessage = originalFailure.getMessage() == null ? "" : originalFailure.getMessage();
            boolean isErrorCode126 = rawFailureMessage.contains("error 126");
            if (isErrorCode126) {
                LOG.warn("Windows error 126 indicates a missing DLL dependency of onnxruntime_providers_cuda.dll, " +
                        "not a missing CUDA install. Typical causes: cuDNN split DLLs missing, wrong CUDA/cuDNN major " +
                        "version, or a stale CUDA entry earlier on PATH shadowing the expected one.");
            }

            if (!isWindows) {
                LOG.warn("On Linux, follow https://onnxruntime.ai/docs/install/#cuda-and-cudnn and make sure " +
                        "libcudart.so.12 and libcudnn.so.9 are on LD_LIBRARY_PATH.");
                return;
            }

            String pathEnvironmentVariable = System.getenv("PATH");
            if (pathEnvironmentVariable == null || pathEnvironmentVariable.isEmpty()) {
                LOG.warn("PATH environment variable is empty. CUDA/cuDNN DLLs cannot be located.");
                return;
            }

            List<Path> pathDirectories = parseWindowsPathEntries(pathEnvironmentVariable);
            LOG.warn("PATH contains {} entries. Scanning for CUDA/cuDNN DLLs...", pathDirectories.size());

            List<Path> cudartMatches = findDllOnPath(pathDirectories, "cudart64_");
            List<Path> cudaRuntime12Matches = findExactDllOnPath(pathDirectories, "cudart64_12.dll");
            List<Path> cudnnLauncherMatches = findExactDllOnPath(pathDirectories, "cudnn64_9.dll");

            reportDllScan("cudart64_12.dll (CUDA 12 runtime)", cudaRuntime12Matches);
            reportDllScan("cudnn64_9.dll (cuDNN 9 launcher)", cudnnLauncherMatches);

            if (cudartMatches.size() > 1) {
                LOG.warn("Multiple cudart64_*.dll versions found on PATH. This can cause error 126 if an older " +
                        "version is picked up first:");
                for (Path cudartMatch : cudartMatches) {
                    LOG.warn("  - {}", cudartMatch);
                }
            }

            if (cudaRuntime12Matches.isEmpty()) {
                LOG.warn("cudart64_12.dll not found on PATH. Install CUDA 12.x and add its bin folder " +
                        "(e.g. C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin) to the system PATH.");
            }
            if (cudnnLauncherMatches.isEmpty()) {
                LOG.warn("cudnn64_9.dll not found on PATH. Install cuDNN 9.x and add its x64 folder " +
                        "(e.g. C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\12.x\\x64) to the system PATH.");
            } else {
                checkCudnnSplitDlls(cudnnLauncherMatches.get(0).getParent());
            }

            LOG.warn("If cudart64_12.dll and cudnn64_9.dll both resolve above but CUDA still fails, check for " +
                    "stale CUDA 11.x or cuDNN 8.x folders on PATH, verify your NVIDIA driver supports CUDA 12.x, " +
                    "and ensure the launcher inherits PATH (log out/in or reboot after editing system PATH).");
        } catch (Throwable diagnosticsFailure) {
            LOG.warn("Failed to collect CUDA load diagnostics: {}", diagnosticsFailure.toString());
        }
    }

    /**
     * Splits {@code PATH} into directory {@link Path}s, tolerating malformed entries and stripping quotes.
     */
    private static List<Path> parseWindowsPathEntries(String pathEnvironmentVariable) {
        List<Path> parsedDirectories = new ArrayList<>();
        for (String rawEntry : pathEnvironmentVariable.split(";")) {
            String trimmedEntry = rawEntry.trim();
            if (trimmedEntry.isEmpty()) continue;
            if (trimmedEntry.startsWith("\"") && trimmedEntry.endsWith("\"") && trimmedEntry.length() >= 2) {
                trimmedEntry = trimmedEntry.substring(1, trimmedEntry.length() - 1);
            }
            try {
                parsedDirectories.add(Paths.get(trimmedEntry));
            } catch (Exception ignored) {
            }
        }
        return parsedDirectories;
    }

    /**
     * Returns all PATH directories that contain a DLL whose filename starts with {@code dllPrefix}
     * and ends with {@code .dll}. Used to detect multiple version-suffixed copies (e.g. cudart64_11, cudart64_12).
     */
    private static List<Path> findDllOnPath(List<Path> pathDirectories, String dllPrefix) {
        List<Path> matches = new ArrayList<>();
        for (Path pathDirectory : pathDirectories) {
            try {
                if (!Files.isDirectory(pathDirectory)) continue;
                try (var directoryStream = Files.newDirectoryStream(pathDirectory, dllPrefix + "*.dll")) {
                    for (Path candidate : directoryStream) {
                        matches.add(candidate);
                    }
                }
            } catch (Exception ignored) {
            }
        }
        return matches;
    }

    /**
     * Returns every PATH directory that contains {@code exactDllName}. The first entry is the one
     * the Windows loader would pick; additional entries indicate shadowing.
     */
    private static List<Path> findExactDllOnPath(List<Path> pathDirectories, String exactDllName) {
        List<Path> matches = new ArrayList<>();
        for (Path pathDirectory : pathDirectories) {
            try {
                Path candidate = pathDirectory.resolve(exactDllName);
                if (Files.isRegularFile(candidate)) {
                    matches.add(candidate);
                }
            } catch (Exception ignored) {
            }
        }
        return matches;
    }

    private static void reportDllScan(String dllLabel, List<Path> matches) {
        if (matches.isEmpty()) {
            LOG.warn("  [MISSING] {}: not found on PATH", dllLabel);
            return;
        }
        LOG.warn("  [OK]      {}: {}", dllLabel, matches.get(0));
        for (int shadowIndex = 1; shadowIndex < matches.size(); shadowIndex++) {
            LOG.warn("            (shadowed by earlier PATH entry) {}", matches.get(shadowIndex));
        }
    }

    /**
     * cuDNN 9 is split across multiple DLLs; only shipping {@code cudnn64_9.dll} causes error 126
     * when the CUDA provider tries to use a cuDNN feature implemented in one of the sub-libraries.
     */
    private static void checkCudnnSplitDlls(Path cudnnLauncherDirectory) {
        if (cudnnLauncherDirectory == null) return;
        String[] requiredCudnnSplitDlls = {
                "cudnn_graph64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_cnn64_9.dll",
                "cudnn_adv64_9.dll",
                "cudnn_heuristic64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll"
        };
        List<String> missingSplitDlls = new ArrayList<>();
        for (String requiredSplitDll : requiredCudnnSplitDlls) {
            if (!Files.isRegularFile(cudnnLauncherDirectory.resolve(requiredSplitDll))) {
                missingSplitDlls.add(requiredSplitDll);
            }
        }
        if (!missingSplitDlls.isEmpty()) {
            LOG.warn("cuDNN 9 directory {} is missing split DLL(s) required by ONNX Runtime: {}. " +
                    "Reinstall cuDNN 9 and copy the full bin/<cuda>/x64 folder, not just cudnn64_9.dll.",
                    cudnnLauncherDirectory, missingSplitDlls);
        }
    }

    private static float[] runWithSession(OrtSession session, Object[][] inputs) {
        Map<String, OnnxTensor> feed = new LinkedHashMap<>();
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try {
            for (Object[] inp : inputs) {
                feed.put((String) inp[0],
                        OnnxTensor.createTensor(env, FloatBuffer.wrap((float[]) inp[1]), (long[]) inp[2]));
            }
            try (OrtSession.Result result = session.run(feed)) {
                OnnxTensor output = (OnnxTensor) result.get(0);
                FloatBuffer buf = output.getFloatBuffer();
                float[] out = new float[buf.remaining()];
                buf.get(out);
                return out;
            }
        } catch (OrtException e) {
            throw new RuntimeException("ONNX inference failed", e);
        } finally {
            for (OnnxTensor t : feed.values()) t.close();
        }
    }

    @Override
    public void close() {
        synchronized (GPU_SLOT_LOCK) {
            if (gpuSlotHolder == this && activeGpuSession != null) {
                try { activeGpuSession.close(); } catch (OrtException ignored) {}
                activeGpuSession = null;
                gpuSlotHolder = null;
            }
        }
        if (cpuSession != null) {
            try { cpuSession.close(); } catch (OrtException ignored) {}
            cpuSession = null;
        }
        if (gpuSession != null) {
            try { gpuSession.close(); } catch (OrtException ignored) {}
            gpuSession = null;
        }
    }
}
