import SwiftSyntax
import SwiftParser
import Foundation

let inputPath = "../input.swift"
let outputPath = "../output.json"

// ✅ 파일 내용을 문자열로 읽어야 함
let sourceCode = try String(contentsOfFile: inputPath)

class Collector: SyntaxVisitor {
    var list: [[String: Any]] = []
    var converter: SourceLocationConverter!

    init(tree: SourceFileSyntax) {
        converter = SourceLocationConverter(fileName: "", tree: tree)
        super.init(viewMode: .sourceAccurate)
        walk(tree)
    }

    override func visit(_ node: DeclReferenceExprSyntax) -> SyntaxVisitorContinueKind {
        let name = node.baseName.text
        let line = converter.location(for: node.positionAfterSkippingLeadingTrivia).line
        list.append(["ident": name, "line": line])
        return .visitChildren
    }
}

do {
    // ✅ 변경된 방식: 문자열 파싱
    let tree = try Parser.parse(source: sourceCode)
    let collector = Collector(tree: tree)

    let jsonData = try JSONSerialization.data(withJSONObject: collector.list, options: [.prettyPrinted])
    try jsonData.write(to: URL(fileURLWithPath: outputPath))
    print("✅ output.json 저장 완료")
} catch {
    print("❌ 오류 발생: \(error)")
    exit(1)
}
