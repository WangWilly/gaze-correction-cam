//
//  CameraOffsetView.swift
//  SettingsApp
//
//  Camera offset controls view for adjusting X, Y, Z position.
//

import SwiftUI

/// Camera offset controls view
struct CameraOffsetView: View {
    
    /// View model
    @ObservedObject var viewModel: SettingsViewModel
    
    /// Step size for offset adjustment
    private let stepSize: Float = 0.5
    
    var body: some View {
        VStack(spacing: 20) {
            // Camera offset section
            GroupBox("Camera Offset (cm)") {
                VStack(spacing: 16) {
                    // X offset (horizontal)
                    offsetSlider(
                        label: "X (Horizontal)",
                        value: $viewModel.cameraOffsetX,
                        range: -20...20,
                        icon: "arrow.left.and.right"
                    )
                    
                    // Y offset (vertical)
                    offsetSlider(
                        label: "Y (Vertical)",
                        value: $viewModel.cameraOffsetY,
                        range: -40...10,
                        icon: "arrow.up.and.down"
                    )
                    
                    // Z offset (depth)
                    offsetSlider(
                        label: "Z (Depth)",
                        value: $viewModel.cameraOffsetZ,
                        range: -20...5,
                        icon: "arrow.up.forward.and.arrow.down.backward"
                    )
                }
                .padding(.vertical, 8)
            }
            
            // Camera parameters section
            GroupBox("Camera Parameters") {
                VStack(spacing: 16) {
                    // Focal length
                    parameterSlider(
                        label: "Focal Length (px)",
                        value: $viewModel.focalLength,
                        range: 300...1200,
                        icon: "camera.aperture"
                    )
                    
                    // IPD
                    parameterSlider(
                        label: "IPD (cm)",
                        value: $viewModel.ipd,
                        range: 4...8,
                        icon: "eye"
                    )
                }
                .padding(.vertical, 8)
            }
            
            // Quick adjustment buttons
            GroupBox("Quick Adjust") {
                HStack(spacing: 12) {
                    // Direction buttons for XY
                    VStack(spacing: 4) {
                        Button(action: { viewModel.cameraOffsetY -= stepSize }) {
                            Image(systemName: "arrow.up")
                                .frame(width: 32, height: 32)
                        }
                        
                        HStack(spacing: 4) {
                            Button(action: { viewModel.cameraOffsetX -= stepSize }) {
                                Image(systemName: "arrow.left")
                                    .frame(width: 32, height: 32)
                            }
                            
                            Button(action: { viewModel.cameraOffsetX = 0; viewModel.cameraOffsetY = -21 }) {
                                Image(systemName: "circle")
                                    .frame(width: 32, height: 32)
                            }
                            
                            Button(action: { viewModel.cameraOffsetX += stepSize }) {
                                Image(systemName: "arrow.right")
                                    .frame(width: 32, height: 32)
                            }
                        }
                        
                        Button(action: { viewModel.cameraOffsetY += stepSize }) {
                            Image(systemName: "arrow.down")
                                .frame(width: 32, height: 32)
                        }
                    }
                    
                    Divider()
                    
                    // Z adjustment
                    VStack(spacing: 8) {
                        Text("Depth")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Button(action: { viewModel.cameraOffsetZ += stepSize }) {
                            Image(systemName: "plus")
                                .frame(width: 32, height: 32)
                        }
                        
                        Text(String(format: "%.1f", viewModel.cameraOffsetZ))
                            .font(.caption.monospacedDigit())
                        
                        Button(action: { viewModel.cameraOffsetZ -= stepSize }) {
                            Image(systemName: "minus")
                                .frame(width: 32, height: 32)
                        }
                    }
                }
                .buttonStyle(.bordered)
                .padding(.vertical, 8)
            }
            
            Spacer()
        }
        .padding()
    }
    
    /// Offset slider with label and value display
    private func offsetSlider(
        label: String,
        value: Binding<Float>,
        range: ClosedRange<Float>,
        icon: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.secondary)
                    .frame(width: 20)
                
                Text(label)
                    .font(.caption)
                
                Spacer()
                
                Text(String(format: "%+.1f cm", value.wrappedValue))
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.secondary)
            }
            
            Slider(value: value, in: range, step: 0.5)
        }
    }
    
    /// Parameter slider with label and value display
    private func parameterSlider(
        label: String,
        value: Binding<Float>,
        range: ClosedRange<Float>,
        icon: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.secondary)
                    .frame(width: 20)
                
                Text(label)
                    .font(.caption)
                
                Spacer()
                
                Text(String(format: "%.1f", value.wrappedValue))
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.secondary)
            }
            
            Slider(value: value, in: range)
        }
    }
}

// MARK: - Preview

struct CameraOffsetView_Previews: PreviewProvider {
    static var previews: some View {
        CameraOffsetView(viewModel: SettingsViewModel())
    }
}
