// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "swift_syntax_runner",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-syntax.git", exact: "510.0.0")
    ],
    targets: [
        .executableTarget(
            name: "swift_syntax_runner",
            dependencies: [
                .product(name: "SwiftSyntax", package: "swift-syntax"),
                .product(name: "SwiftParser", package: "swift-syntax")  // ✅ 이걸 사용해야 함
            ]
        )
    ]
)
