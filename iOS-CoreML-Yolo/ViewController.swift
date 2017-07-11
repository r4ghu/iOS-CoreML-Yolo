//
//  ViewController.swift
//  iOS-CoreML-Yolo
//
//  Created by Sri Raghu Malireddi on 16/06/17.
//  Copyright © 2017 Sri Raghu Malireddi. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML

class ViewController: UIViewController {

    @IBOutlet weak var previewView: PreviewView!
    @IBOutlet weak var imageView: UIImageView!
    
    
    // Session - Initialization
    private let session = AVCaptureSession()
    private var isSessionRunning = false
    private let sessionQueue = DispatchQueue(label: "session queue", attributes: [], target: nil)
    private var permissionGranted = false
    
    // Model
    let model = TinyYOLOv1() // In: Image<RGB,448,448> Out: MultiArray<Double,1470>
    let modelInputSize = CGSize(width: 448, height: 448)
    
    //Output Results
    let frame = CGRect(origin: CGPoint(x: 0, y: 0), size: CGSize(width: 448, height: 448))
    var cgImage: CGImage?
    var uiImage:UIImage?
    
    // Yolo variables
    let S = 7
    let B = 2
    let C = 20
    let pNorm = 1.8
    let threshold = 0.17 //0.17
    let num_grid_cells = 49 //SS
    let class_probabilities_size = 980 //prob_size
    let conf_grid_cell_size = 98 //conf_size
    let car_class = 6
    
    var lastTimestamp = CMTime()
    
    struct Box {
        let x : Double
        let y : Double
        let width : Double
        let height : Double
        let objClass : Double
        var probability : Double
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        // Set some features for PreviewView
        previewView.videoPreviewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill
        previewView.session = session
        
        // Check for permissions
        checkPermission()
        
        // Configure Session in session queue
        sessionQueue.async { [unowned self] in
            self.configureSession()
        }
        
        cgImage = CIContext().createCGImage(CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)), from: frame)!
        uiImage = UIImage(cgImage: cgImage!)
    }
    
    // Check for camera permissions
    private func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(forMediaType: AVMediaTypeVideo) {
        case .authorized:
            permissionGranted = true
        case .notDetermined:
            requestPermission()
        default:
            permissionGranted = false
        }
    }
    
    // Request permission if not given
    private func requestPermission() {
        sessionQueue.suspend()
        AVCaptureDevice.requestAccess(forMediaType: AVMediaTypeVideo) { [unowned self] granted in
            self.permissionGranted = granted
            self.sessionQueue.resume()
        }
    }
    
    // Start session
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        sessionQueue.async {
            self.session.startRunning()
            self.isSessionRunning = self.session.isRunning
        }
    }
    
    // Stop session
    override func viewWillDisappear(_ animated: Bool) {
        sessionQueue.async { [unowned self] in
            if self.permissionGranted {
                self.session.stopRunning()
                self.isSessionRunning = self.session.isRunning
            }
        }
        super.viewWillDisappear(animated)
    }
    
    // Configure session properties
    private func configureSession() {
        guard permissionGranted else { return }
        
        session.beginConfiguration()
        session.sessionPreset = AVCaptureSessionPreset1280x720
        
        guard let captureDevice = AVCaptureDevice.defaultDevice(withDeviceType: .builtInWideAngleCamera, mediaType: AVMediaTypeVideo, position: .back) else { return }
        guard let captureDeviceInput = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        guard session.canAddInput(captureDeviceInput) else { return }
        session.addInput(captureDeviceInput)
        
        let videoOutput = AVCaptureVideoDataOutput()
        
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "sample buffer"))
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String : kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        guard session.canAddOutput(videoOutput) else { return }
        session.addOutput(videoOutput)
        
        guard let connection = videoOutput.connection(withMediaType: AVMediaTypeVideo) else {return}
        guard connection.isVideoOrientationSupported else { return }
        connection.videoOrientation = .portrait
        
        session.commitConfiguration()
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
    }
    
    func output2Box(_ output: MLMultiArray) -> Array<Box> {
        var class_probabilities = Array<Array<Double>>()
        var conf_grid_cell = Array<Array<Double>>()
        var box_coords = Array<Array<Array<Double>>>()
        var Boxes = Array<Box>()
        
        let queue:OperationQueue = OperationQueue()
        queue.name = "com.sri.iOS.iOS-CoreML-Yolo.output"
        queue.qualityOfService = QualityOfService.default
        queue.maxConcurrentOperationCount = 6
        
        let operation01:BlockOperation = BlockOperation {
            var counter = 0
            for _ in 0..<self.num_grid_cells {
                var class_probabilities_row = Array<Double>()
                for _ in 0..<self.C {
                    class_probabilities_row.append(Double(output[counter]))
                    counter += 1
                }
                class_probabilities.append(class_probabilities_row)
            }
            //print(counter) //980!
        }
        
        let operation02:BlockOperation = BlockOperation {
            var counterOp1 = self.class_probabilities_size
            for _ in 0..<self.num_grid_cells {
                var conf_grid_cell_row = Array<Double>()
                for _ in 0..<self.B {
                    conf_grid_cell_row.append(Double(output[counterOp1]))
                    counterOp1 += 1
                }
                conf_grid_cell.append(conf_grid_cell_row)
            }
            //print(counterOp1) //1078!
        }
        
        let operation03:BlockOperation = BlockOperation{
            var counterOp2 = self.class_probabilities_size + self.conf_grid_cell_size
            for _ in 0..<self.num_grid_cells {
                var box_coords_channel = Array<Array<Double>>()
                for _ in 0..<self.B {
                    var xywh = Array<Double>()
                    for _ in 0..<4 {
                        xywh.append(Double(output[counterOp2]))
                        counterOp2 += 1
                    }
                    box_coords_channel.append(xywh)
                }
                box_coords.append(box_coords_channel)
            }
            //print(counterOp2) //1470!
        }
        
        queue.addOperations([operation01,operation02,operation03], waitUntilFinished: true)
        
        for grid in 0..<num_grid_cells {
            for b in 0..<B {
                //print(class_probabilities[grid][car_class] * conf_grid_cell[grid][b])
                if class_probabilities[grid][car_class] * conf_grid_cell[grid][b] >= threshold {
                    let box = Box(x: (box_coords[grid][b][0] + Double(grid).truncatingRemainder(dividingBy: Double(S))) / Double(S),
                                  y: (box_coords[grid][b][1] + floor(Double(grid)/Double(S))) / Double(S),
                                  width: pow(box_coords[grid][b][2], pNorm),
                                  height: pow(box_coords[grid][b][3], pNorm),
                                  objClass: conf_grid_cell[grid][b],
                                  probability: class_probabilities[grid][car_class] * conf_grid_cell[grid][b])
                    Boxes.append(box)
                }
            }
        }
        Boxes.sort { $0.probability > $1.probability }
        for i in 0..<Boxes.count {
            if Boxes[i].probability == 0 {
                continue
            }
            for j in (i+1)..<Boxes.count {
                if box_iou(a: Boxes[i], b: Boxes[j]) >= 0.4 {
                    Boxes[j].probability = 0
                }
            }
        }
        Boxes = Boxes.filter{$0.probability > 0}
        //print(Boxes.count)
        return Boxes
    }
    
    func overlap(x1 : Double, w1 : Double, x2 : Double, w2 : Double) -> Double {
        let l1 = x1 - w1 / 2.0
        let l2 = x2 - w2 / 2.0
        let left = max(l1,l2)
        let r1 = x1 + w1 / 2.0
        let r2 = x2 + w2 / 2.0
        let right = min(r1,r2)
        return right - left
    }
    
    func box_intersection(a: Box, b: Box) -> Double {
        let w = overlap(x1: a.x, w1: a.width, x2: b.x, w2: b.width)
        let h = overlap(x1: a.y, w1: a.height, x2: b.y, w2: b.height)
        if w<0 || h<0 {
            return 0
        }
        return w * h
    }
    
    func box_iou(a: Box, b: Box) -> Double {
        let i = box_intersection(a: a, b: b)
        return i / (a.width * a.height + b.width * b.height - i)
    }
    
    func drawRectangleOnImage(image: UIImage, boxes: Array<Box>) -> UIImage {
        let imageSize = image.size
        let scale: CGFloat = 0
        UIGraphicsBeginImageContextWithOptions(imageSize, false, scale)
        
        image.draw(at: CGPoint.zero)
        UIColor(red: 0.0, green: 1.0, blue: 1.0, alpha: 1.0).set()
        for box in boxes {
            let rectangle = CGRect(x: (box.x - box.width/2.0)*Double(modelInputSize.width),
                                   y: (box.y - box.height/2.0)*Double(modelInputSize.height),
                                   width: box.width * Double(modelInputSize.width),
                                   height: box.height * Double(modelInputSize.height))
            UIRectFrame(rectangle)
        }
        
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage!
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        
        guard let pixelBuffer = UIImage(ciImage: ciImage).resize(modelInputSize)?.pixelBuffer() else { return }
        let start = NSDate()
        let output = try? model.prediction(image: pixelBuffer)
        let end = NSDate()
        // Post-processing
        let boxes = output2Box((output?.output)!)
        //let ciimage = CIImage(cvPixelBuffer: pixelBuffer)
        
        let imageS = self.drawRectangleOnImage(image: self.uiImage!, boxes: boxes)
        DispatchQueue.main.async {
            self.imageView.image = imageS
            print(1/end.timeIntervalSince(start as Date))
        }
    }
    
}

extension UIImage {
    func resize(_ size: CGSize)-> UIImage? {
        UIGraphicsBeginImageContext(size)
        draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }
    
    func pixelBuffer() -> CVPixelBuffer? {
        let width = self.size.width
        let height = self.size.height
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(width),
                                         Int(height),
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)
        
        guard let resultPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(resultPixelBuffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(width),
                                      height: Int(height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(resultPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                                        return nil
        }
        
        context.translateBy(x: 0, y: height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return resultPixelBuffer
    }
    
    
}

