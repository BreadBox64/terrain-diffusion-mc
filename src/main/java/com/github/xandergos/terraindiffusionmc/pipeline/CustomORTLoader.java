package com.github.xandergos.terraindiffusionmc.pipeline;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class CustomORTLoader {
	static {
		try {
			Path dir = Files.createTempDirectory("ort-natives");
			copy("/natives/linux-x86_64/libonnxruntime.so", dir);
			copy("/natives/linux-x86_64/libonnxruntime_providers_shared.so", dir);
			//copy("/natives/linux-x86_64/libonnxruntime_providers_migraphx.so", dir);
			copy("/natives/linux-x86_64/libonnxruntime4j_jni.so", dir);

			System.load(dir.resolve("libonnxruntime.so").toString());
			System.load(dir.resolve("libonnxruntime_providers_shared.so").toString());
			//System.load(dir.resolve("libonnxruntime_providers_migraphx.so").toString());
			System.setProperty("onnxruntime.native.path", dir.toAbsolutePath().toString());
		} catch (Exception e) {
			throw new RuntimeException("Failed to load custom ORT natives", e);
		}
	}

	private static void copy(String resource, Path dir) throws IOException {
		try (InputStream in = CustomORTLoader.class.getResourceAsStream(resource)) {
			Files.copy(in, dir.resolve(Paths.get(resource).getFileName()));
		}
	}
}