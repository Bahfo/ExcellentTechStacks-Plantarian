require("dotenv").config({ path: require("path").join(__dirname, "../.env") });
const { MongoClient } = require("mongodb");

const mongoUri = process.env.MONGO_URI || "mongodb://localhost:27017";
const dbName = process.env.DB_NAME || "plantarian_db";

const client = new MongoClient(mongoUri);

async function runEDA() {
    try {
        await client.connect();
        const db = client.db(dbName);
        const col = db.collection("images_metadata");
        console.log(` Connected to MongoDB: ${dbName}`);

        const total = await col.countDocuments();
        if (total === 0) {
            console.log("No documents found in 'images_metadata'.");
            await client.close();
            return;
        }
        console.log(`TOTAL IMAGES IN DATABASE: ${total}\n`);

        const classDist = await col.aggregate([
            {
                $group: {
                    _id: "$datasetClass",
                    count: { $sum: 1 }
                }
            },
            { $sort: { count: -1 } }
        ]).toArray();

        console.log("TOP 10 INDIVIDUAL FOLDER CLASSES");
        console.table(
            classDist.slice(0, 10).map((item, idx) => ({
                Rank: idx + 1,
                "Folder Class": item._id,
                "Image Count": item.count,
                "Percentage (%)": ((item.count / total) * 100).toFixed(2)
            }))
        );

        const speciesDist = await col.aggregate([
            {
                $group: {
                    _id: "$species",
                    count: { $sum: 1 }
                }
            },
            { $sort: { count: -1 } }
        ]).toArray();

        console.log("\nPLANT SPECIES DISTRIBUTION");
        console.table(
            speciesDist.map((item) => ({
                "Plant Species": item._id,
                "Total Images": item.count,
                "Percentage (%)": ((item.count / total) * 100).toFixed(2)
            }))
        );

        const healthDist = await col.aggregate([
            {
                $group: {
                    _id: "$healthState",
                    count: { $sum: 1 }
                }
            },
            { $sort: { count: -1 } }
        ]).toArray();

        console.log("\nHEALTH STATE BALANCE");
        console.table(
            healthDist.map((item) => ({
                Condition: item._id.toUpperCase(),
                "Image Count": item.count,
                "Percentage (%)": ((item.count / total) * 100).toFixed(2)
            }))
        );

    } catch (error) {
        console.error("An error occurred during EDA execution:", error);
    } finally {
        await client.close();
        console.log("EDA complete. MongoDB Connection Closed");
    }
}

runEDA();