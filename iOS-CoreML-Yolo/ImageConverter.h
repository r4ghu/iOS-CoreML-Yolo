//
//  ImageConverter.h
//  iOS-CoreML-Yolo
//
//  Created by Sri Raghu Malireddi on 16/06/17.
//  Copyright © 2017 Sri Raghu Malireddi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

@interface ImageConverter : NSObject
    
+ (CVPixelBufferRef) pixelBufferFromImage: (CGImageRef) image;
    
@end

