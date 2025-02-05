import sqlite3 from 'sqlite3'

const db = new sqlite3.Database(':memory:');

db.serialize(() => {
    db.run("CREATE TABLE lorem (info TEXT)");

    const stmt = db.prepare("INSERT INTO lorem VALUES (?)");
    for (let i = 0; i < 10; i++) {
        stmt.run("Ipsum " + i);
    }
    stmt.finalize();
});

/**
 * @param {FastifyInstance} fastify
 * @param {Object} options
 */
async function dbConnector (fastify, options) {
    fastify.get('/greg', async (req, res) => {
        try {
            const rows = await new Promise((resolve, reject) => {
                db.all("SELECT rowid AS id, info FROM lorem", (err, rows) => {
                    if (err)
                        reject(err);
                    else
                        resolve(rows);
                });
            });

            res.send(JSON.stringify(rows));
        } catch (error) {
            res.status(500).send("Database error");
        }
    });
    // fastify.register(fastifyMongo, {
    //     url: 'mongodb://localhost:27017/test_database'
    // })
}

// Wrapping a plugin function with fastify-plugin exposes the decorators
// and hooks, declared inside the plugin to the parent scope.
export default dbConnector
