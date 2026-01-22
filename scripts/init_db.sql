-- database schema for CloudSeg

CREATE TABLE IF NOT EXISTS segmentation_jobs (
    id SERIAL PRIMARY KEY,

    image_url TEXT NOT NULL,
    mask_url TEXT,

    model_name VARCHAR(100) NOT NULL DEFAULT 'deeplabv3_resnet50',

    status VARCHAR(20) NOT NULL DEFAULT 'processing',
    inference_time_ms INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- index for faster queries (optional)
CREATE INDEX IF NOT EXISTS idx_segmentation_created_at
ON segmentation_jobs (created_at);
