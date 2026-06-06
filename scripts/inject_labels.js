const path = require("path");
require("dotenv").config({path: path.join(__dirname, "../.env")});

const { MongoClient } = require("mongodb");
const client = new MongoClient(process.env.MONGO_URI);

async function injectLabels() {
    await client.connect();
    const db = client.db(process.env.DB_NAME);
    const col = db.collection("images_metadata");

    const speciesList = await col.distinct("species");
    const classList = await col.distinct("datasetClass");

    speciesList.sort();
    classList.sort();

    const speciesMap = {};
    const classMap = {};

    speciesList.forEach((s, i) => { speciesMap[s] = i; });

    classList.forEach((c, i) => { classMap[c] = i; });

    console.log("Species:", speciesList.length);
    console.log("Classes:", classList.length);
    const cursor = col.find({});

    let batch = [];
    while (await cursor.hasNext()) {
        const doc = await cursor.next();

        batch.push({
            updateOne: {
                filter: { _id: doc._id },
                update: {
                    $set: {
                        speciesLabel:
                            speciesMap[doc.species],
                        classLabel:
                            classMap[doc.datasetClass]
                    }
                }
            }
        });

        if (batch.length === 1000) {
            await col.bulkWrite(batch);
            batch = [];
        }
    }

    if (batch.length > 0) await col.bulkWrite(batch);
    await client.close();
    console.log("Label injection completed.");
}

injectLabels().catch(console.error);