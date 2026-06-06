const fs = require("fs-extra");
const mime = require("mime-types");
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, "../.env") });

const {MongoClient,GridFSBucket} = require("mongodb");
const DATASET_DIR = path.join(__dirname,"../dataset/raw/color");
const client = new MongoClient(process.env.MONGO_URI);

async function ingestDataset() {
    await client.connect();
    console.log("MongoDB connected.");

    const database = client.db(process.env.DB_NAME);
    const metadataCollection = database.collection("images_metadata");
    const bucket = new GridFSBucket(database,{bucketName: "plant_images"});

    await metadataCollection.createIndex({species: 1});
    await metadataCollection.createIndex({disease: 1});
    await metadataCollection.createIndex({healthState: 1});
    await metadataCollection.createIndex({datasetClass: 1});

    const folders = await fs.readdir(DATASET_DIR);

    let totalImages = 0;
    for (const folder of folders) {
        const folderPath = path.join(DATASET_DIR, folder);
        const stat = await fs.stat(folderPath);
        if (!stat.isDirectory()) continue;

        const [species, disease] = folder.split("___");
        const healthState = disease.toLowerCase() === "healthy"
                ? "healthy"
                : "diseased";

        const files = await fs.readdir(folderPath);
        for (const file of files) { 
            const extension = path.extname(file).toLowerCase();
            if (![".jpg",".jpeg",".png"].includes(extension)) continue;

            const filePath = path.join(folderPath, file);
            const readStream = fs.createReadStream(filePath);
            const uploadStream = bucket.openUploadStream(file, {
                contentType: mime.lookup(filePath)
            });

            await new Promise(
                (resolve, reject) => {
                    readStream
                        .pipe(uploadStream)
                        .on("error", reject)
                        .on("finish", resolve);
                }
            );

            const metadata = {species, disease, healthState,
                datasetClass: folder,
                imageType: "color",
                originalFilename: file,
                sourcePath: filePath,
                gridfsFileId: uploadStream.id,
                uploadedAt: new Date()
            };

            await metadataCollection.insertOne(metadata);
            totalImages++;

            if (totalImages % 100 === 0) {
                console.log(`Uploaded ${totalImages} images`);
            }
        }
    }

    console.log(`Finished Uploading ${totalImages} Images`);

    await client.close();
    console.log(
        "MongoDB Connection Closed"
    );
}

ingestDataset().catch(console.error);