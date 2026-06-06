const fs = require("fs");
const path = require("path");
const { MongoClient, GridFSBucket } = require("mongodb");
require("dotenv").config({ path: path.join(__dirname, "../.env") });

async function exportForPython() {
    const client = new MongoClient(process.env.MONGO_URI);
    await client.connect();
    const db = client.db(process.env.DB_NAME);
    
    const MIN_SIZE = 10240; // 10 KB
    const exportDir = path.resolve(__dirname, "../staging_data");
    if (!fs.existsSync(exportDir)) fs.mkdirSync(exportDir);

    console.log("Extracting clean data for Python pipeline...");
    const pipeline = [
        { $lookup: { from: "plant_images.files", 
            localField: "gridfsFileId", foreignField: "_id", as: "file" } },
        { $unwind: "$file" },
        { $match: { "file.length": { $gte: MIN_SIZE } } },
        { $project: { file: 0 } }
    ];

    const data = await db.collection("images_metadata").aggregate(pipeline).toArray();
    const bucket = new GridFSBucket(db, { bucketName: "plant_images" });
    const metadata_log = [];

    for (let i = 0; i < data.length; i++) {
        const item = data[i];
        const filename = `${item._id}.jpg`;
        
        await new Promise((resolve, reject) => {
            bucket.openDownloadStream(item.gridfsFileId)
                .pipe(fs.createWriteStream(path.join(exportDir, filename)))
                .on("finish", resolve).on("error", reject);
        });

        metadata_log.push({
            filename: filename,
            classLabel: item.classLabel,
            speciesLabel: item.speciesLabel,
            datasetSplit: item.datasetSplit
        });
        if (i % 1000 === 0) console.log(`Exported ${i} / ${data.length}`);
    }

    fs.writeFileSync(path.join(exportDir, "metadata.json"), JSON.stringify(metadata_log));
    console.log("JS Extraction Complete. Ready for Python.");
    await client.close();
}

exportForPython();