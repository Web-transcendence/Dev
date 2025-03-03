import fastify, { FastifyRequest} from 'fastify'

interface IParam {
    id: string;
}

const app = fastify();

app.get('/user-management/:id', async (req: FastifyRequest<{ Params: IParam }>) => {
    console.log("user test");

    return {
        id: req.params.id,
        name: "tes tname"
    };
});

app.listen({port: 8000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})