import fastify from 'fastify'
import socialRoutes from "./routes.js"

const app = fastify();

app.register(socialRoutes);

app.listen({port: 6000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})