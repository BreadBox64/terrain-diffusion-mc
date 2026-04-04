package com.github.xandergos.terraindiffusionmc.mixin.client;

import com.github.xandergos.terraindiffusionmc.client.WorldScaleSettingsScreen;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.screen.world.CreateWorldScreen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.text.Text;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Adds Terrain Diffusion world settings access to the world creation screen.
 */
@Mixin(CreateWorldScreen.class)
public abstract class CreateWorldScreenMixin extends Screen {
    @Unique
    private static final int TERRAIN_SETTINGS_BUTTON_WIDTH = 160;
    @Unique
    private static final int TERRAIN_SETTINGS_BUTTON_HEIGHT = 20;

    protected CreateWorldScreenMixin(Text title) {
        super(title);
    }

    @Inject(method = "init", at = @At("TAIL"))
    private void terrainDiffusionMc$addTerrainSettingsButton(CallbackInfo callbackInfo) {
        int xPosition = 8;
        int yPosition = this.height - TERRAIN_SETTINGS_BUTTON_HEIGHT - 8;
        this.addDrawableChild(ButtonWidget.builder(
                        Text.translatable("terrain-diffusion-mc.world_settings.open_button"),
                        buttonWidget -> {
                            if (this.client != null) {
                                this.client.setScreen(new WorldScaleSettingsScreen((CreateWorldScreen) (Object) this));
                            }
                        })
                .dimensions(xPosition, yPosition, TERRAIN_SETTINGS_BUTTON_WIDTH, TERRAIN_SETTINGS_BUTTON_HEIGHT)
                .build());
    }
}
