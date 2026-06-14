// .NET 8 Minimal API in front of the Python classification service.
// Handles request validation and forwarding through a typed HttpClient; the ML
// stays in Python.

using System.ComponentModel.DataAnnotations;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// Base URL of the Python FastAPI service (configurable via env / appsettings).
var pythonApiBase = builder.Configuration["PYTHON_API_BASE"]
    ?? Environment.GetEnvironmentVariable("PYTHON_API_BASE")
    ?? "http://localhost:8000";

builder.Services.AddHttpClient("python", client =>
{
    client.BaseAddress = new Uri(pythonApiBase);
    client.Timeout = TimeSpan.FromSeconds(30);
});

var app = builder.Build();

app.MapGet("/health", () => Results.Ok(new { status = "ok", upstream = pythonApiBase }));

app.MapPost("/classify", async (ClassifyRequest req, IHttpClientFactory factory, CancellationToken ct) =>
{
    var validation = new List<ValidationResult>();
    if (!Validator.TryValidateObject(req, new ValidationContext(req), validation, true))
        return Results.ValidationProblem(validation.ToDictionary(v => v.MemberNames.FirstOrDefault() ?? "", v => new[] { v.ErrorMessage ?? "" }));

    var client = factory.CreateClient("python");
    try
    {
        using var response = await client.PostAsJsonAsync("/classify", req, ct);
        if (!response.IsSuccessStatusCode)
        {
            var detail = await response.Content.ReadAsStringAsync(ct);
            return Results.Problem($"Upstream returned {(int)response.StatusCode}: {detail}",
                statusCode: StatusCodes.Status502BadGateway);
        }

        var payload = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
        return Results.Ok(payload);
    }
    catch (HttpRequestException ex)
    {
        return Results.Problem($"Python service unreachable at {pythonApiBase}: {ex.Message}",
            statusCode: StatusCodes.Status503ServiceUnavailable);
    }
});

app.Run();

// Mirrors the Python ClassifyRequest contract.
public record ClassifyRequest(
    [property: Required, MinLength(1)] string Text,
    [property: RegularExpression("^(encoder|llm)$")] string Method = "encoder");
